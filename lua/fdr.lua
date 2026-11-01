-- Face Reframe bridge for Resolve Free 21.1+: Lua runs the Resolve API,
-- Python runs the heavy compute (face tracking, whisper, audio) as a
-- subprocess exchanging a plain-text job file and a Lua-table result file.

local M = {}

local function log(msg)
    print(msg)
end

local function _int(v, d)
    local n = tonumber(v)
    if n == nil then return d or 0 end
    return math.floor(n + 0.5)
end

local function _float(v, d)
    local n = tonumber(v)
    if n == nil then return d or 0.0 end
    return n
end

local function file_exists(path)
    if not path or path == "" then return false end
    local fh = io.open(path, "r")
    if fh then fh:close() return true end
    return false
end

local function timeline_size(timeline)
    return _int(timeline:GetSetting("timelineResolutionWidth"), 1920),
           _int(timeline:GetSetting("timelineResolutionHeight"), 1080)
end

local function make_timeline_vertical(timeline, aspect, height)
    local width, h = 0, 0
    if aspect <= 1 then
        width = math.floor(height * aspect / 2 + 0.5) * 2
        h = height
    else
        width = height
        h = math.floor(width / aspect / 2 + 0.5) * 2
    end
    local w0, h0 = timeline_size(timeline)
    if w0 == width and h0 == h then return end
    timeline:SetSetting("useCustomSettings", "1")
    local okw = timeline:SetSetting("timelineResolutionWidth", tostring(width))
    local okh = timeline:SetSetting("timelineResolutionHeight", tostring(h))
    if okw and okh then
        log("Timeline resolution set to " .. width .. "x" .. h)
    else
        log("Could not change timeline resolution; set it to " .. width .. "x" .. h ..
            " in Timeline > Timeline Settings")
    end
end

local function target_items(timeline, scope, track_index)
    if scope == "track" then
        return timeline:GetItemListInTrack("video", track_index) or {}
    end
    if scope == "all" then
        local items = {}
        for t = 1, _int(timeline:GetTrackCount("video")) do
            local tr = timeline:GetItemListInTrack("video", t)
            if tr then
                for _, it in ipairs(tr) do items[#items + 1] = it end
            end
        end
        return items
    end
    local it = timeline:GetCurrentVideoItem()
    if it then return {it} end
    return {}
end

local function source_range(item)
    local start
    if item.GetSourceStartFrame then
        local ok, v = pcall(function() return item:GetSourceStartFrame() end)
        if ok then start = v end
    end
    if start == nil then
        local ok, v = pcall(function() return item:GetLeftOffset() end)
        if ok then start = v end
    end
    return math.max(0, _int(start)), _int(item:GetDuration())
end

local function item_timeline_start(item)
    local ok, v = pcall(function() return item:GetStart() end)
    if ok and v ~= nil then return _float(v) end
    local ok2, e = pcall(function() return item:GetEnd() end)
    if ok2 and e ~= nil then
        return _float(e) - _float(item:GetDuration())
    end
    return 0.0
end

local function collect(ctx)
    local project = ctx.resolve:GetProjectManager():GetCurrentProject()
    local timeline = project and project:GetCurrentTimeline() or nil
    if timeline == nil then
        log("Open a timeline first.")
        return nil
    end
    ctx.project = project
    ctx.timeline = timeline
    ctx.tl_fps = _float(timeline:GetSetting("timelineFrameRate"), 25.0)
    local items = target_items(timeline, ctx.scope or "current", ctx.track_index or 1)
    if #items == 0 then
        log("No clip found (put the playhead over a clip, or use an All Tracks script).")
        return nil
    end
    ctx.items = {}
    local id = 0
    for _, item in ipairs(items) do
        local mpi = item:GetMediaPoolItem()
        local path = mpi and mpi:GetClipProperty("File Path") or ""
        if mpi and file_exists(path) then
            id = id + 1
            local start, count = source_range(item)
            local clip_fps = _float(mpi:GetClipProperty("FPS"), ctx.tl_fps)
            if ctx.tl_fps > 0 and clip_fps > 0 and math.abs(clip_fps - ctx.tl_fps) > 0.01 then
                log("Warning: " .. item:GetName() .. " is " .. clip_fps ..
                    " fps on a " .. ctx.tl_fps .. " fps timeline; keys may drift")
            end
            ctx.items[#ctx.items + 1] = {
                id = tostring(id), item = item, mpi = mpi, path = path,
                start = start, count = count, fps = clip_fps,
                tl_start = item_timeline_start(item),
            }
        else
            log("Skipping " .. item:GetName() .. ": source file not found")
        end
    end
    if #ctx.items == 0 then
        log("Nothing to process.")
        return nil
    end
    return ctx
end

local function write_job(ctx, path)
    local fh = io.open(path, "w")
    fh:write("OP\t" .. ctx.op .. "\n")
    fh:write("TLFPS\t" .. ctx.tl_fps .. "\n")
    for k, v in pairs(ctx.job_opts or {}) do
        fh:write(k .. "\t" .. tostring(v) .. "\n")
    end
    for _, it in ipairs(ctx.items) do
        fh:write(table.concat({"ITEM", it.id, it.path, it.start, it.count, it.fps, it.tl_start}, "\t") .. "\n")
    end
    fh:close()
end

local function run_python(ctx)
    local job = os.tmpname()
    write_job(ctx, job)
    local cmd = string.format(
        'env "PYTHONPATH=%s:%s" "%s" -m face_reframe.exec_ops "%s"',
        ctx.lib_dir, ctx.deps_dir, ctx.python, job)
    log("Running compute step (this can take a while on first use)...")
    local ok = os.execute(cmd)
    if ok ~= true and ok ~= 0 then
        log("Compute step failed (exit " .. tostring(ok) .. ") — check that Python and the deps are installed.")
        return nil
    end
    local out = dofile(job .. ".out.lua")
    if type(out) ~= "table" then
        log("Compute step returned no results.")
        return nil
    end
    for _, line in ipairs(out.log or {}) do
        log(line)
    end
    return out
end

-- ==== apply: reframe ====

local MERGE_NAME = "FaceReframeMerge"
local BG_NAME = "FaceReframeCanvas"

local function mark_comp_dirty(comp, tool)
    pcall(function()
        local attrs = comp:GetAttrs() or {}
        local pos = tool:GetAttrs()["TOOLS_LayerOriginX"]
        if pos ~= nil then
            tool:SetAttrs({TOOLS_LayerOriginX = pos})
        end
        tool:SetAttrs({TOOLB_PassThrough = true})
        tool:SetAttrs({TOOLB_PassThrough = false})
    end)
end

local function apply_reframe_item(ctx, it, entry, out_w, out_h)
    local item = it.item
    local comp
    if _int(item:GetFusionCompCount()) > 0 then
        comp = item:GetFusionCompByIndex(1)
    else
        comp = item:AddFusionComp()
    end
    if comp == nil then
        log("Could not create a Fusion comp on " .. item:GetName())
        return
    end
    comp:Lock()
    comp:StartUndo("Face Reframe")
    local merge
    local ok, err = pcall(function()
        local media_out = (comp:GetToolList(false, "MediaOut") or {})[1]
        local media_in = (comp:GetToolList(false, "MediaIn") or {})[1]
        if media_out == nil then error("no MediaOut node in the comp") end
        local attrs = comp:GetAttrs() or {}
        local t0 = _int(attrs["COMPN_RenderStart"], 0)

        merge = comp:FindTool(MERGE_NAME)
        local bg = comp:FindTool(BG_NAME)
        if merge == nil then
            local upstream = media_out.Input:GetConnectedOutput()
            if upstream == nil and media_in ~= nil then
                upstream = media_in.Output
            end
            bg = comp:AddTool("Background", -32768, -32768)
            bg:SetAttrs({TOOLS_Name = BG_NAME})
            merge = comp:AddTool("Merge", -32768, -32768)
            merge:SetAttrs({TOOLS_Name = MERGE_NAME})
            merge.Background = bg.Output
            merge.Foreground = upstream
            media_out.Input = merge.Output
        end
        bg.UseFrameFormatSettings = 0
        bg.Width = out_w
        bg.Height = out_h
        bg.TopLeftRed, bg.TopLeftGreen, bg.TopLeftBlue, bg.TopLeftAlpha = 0, 0, 0, 1

        -- correct the size for a proxy-resolution MediaIn
        local factor = 1.0
        if media_in ~= nil then
            local ok2, img = pcall(function() return media_in.Output[t0] end)
            if ok2 and img ~= nil and img.Width and img.Width > 0 then
                factor = entry.src_w / img.Width
            end
        end

        pcall(function() merge.Size = nil end)
        if entry.size_const then
            merge.Size = entry.size_const * factor
        else
            merge:AddModifier("Size", "BezierSpline")
            for _, k in ipairs(entry.size_keys or {}) do
                merge:SetInput("Size", k.v * factor, t0 + k.f)
            end
        end
        pcall(function() merge.Center = nil end)
        merge:AddModifier("Center", "XYPath")
        for _, k in ipairs(entry.keys) do
            merge:SetInput("Center", {k.x, k.y}, t0 + k.f)
        end
        log("  " .. #entry.keys .. " keyframes on " .. item:GetName())
        for _, w in ipairs(entry.wide or {}) do
            log("  wide shot frames " .. w.a .. "-" .. w.b .. " -> zoom " ..
                string.format("%.2f", w.zoom) .. "x")
        end
    end)
    comp:EndUndo(true)
    comp:Unlock()
    if not ok then
        log("  reframe failed on " .. item:GetName() .. ": " .. tostring(err))
        return
    end
    if merge ~= nil then
        mark_comp_dirty(comp, merge)
    end
end

local function apply_reframe(ctx, out)
    if ctx.set_resolution ~= false then
        make_timeline_vertical(ctx.timeline, ctx.aspect or 0.5625, ctx.output_height or 1920)
    end
    local out_w, out_h = timeline_size(ctx.timeline)
    local done = 0
    for _, it in ipairs(ctx.items) do
        local entry = out.items[it.id]
        if entry then
            apply_reframe_item(ctx, it, entry, out_w, out_h)
            done = done + 1
        end
    end
    log("Done: " .. done .. " clip(s) reframed.")
end

-- ==== apply: cut ====

local function apply_cut(ctx, out)
    local entries = {}
    for _, it in ipairs(ctx.items) do
        local entry = out.items[it.id]
        if entry then
            for _, span in ipairs(entry.spans or {}) do
                if span[2] - span[1] >= 1 then
                    entries[#entries + 1] = {
                        mediaPoolItem = it.mpi,
                        startFrame = span[1],
                        endFrame = span[2],
                    }
                end
            end
        end
    end
    if #entries == 0 then
        log("Nothing kept — check the silence settings.")
        return
    end
    local pool = ctx.project:GetMediaPool()
    local suffix = ctx.new_timeline_suffix or " (no pauses)"
    local name = ctx.timeline:GetName() .. suffix
    local new_tl = pool:CreateEmptyTimeline(name)
    if new_tl == nil then
        log("Could not create a new timeline.")
        return
    end
    ctx.project:SetCurrentTimeline(new_tl)
    for _, key in ipairs({"useCustomSettings", "timelineResolutionWidth",
                          "timelineResolutionHeight", "timelineFrameRate"}) do
        local value = ctx.timeline:GetSetting(key)
        if value then
            new_tl:SetSetting(key, value)
        end
    end
    pool:AppendToTimeline(entries)
    log("Done: " .. #entries .. " segments kept in '" .. name .. "'.")
end

-- ==== apply: subtitles ====

local function apply_subtitles(ctx, out)
    if not out.srt then
        log("No speech found.")
        return
    end
    log("Wrote " .. out.srt .. " (" .. tostring(out.captions or 0) .. " captions)")
    if ctx.timeline.ImportSubtitles then
        if ctx.timeline:ImportSubtitles(out.srt) then
            log("Subtitles imported into the timeline.")
        else
            log("ImportSubtitles failed — import " .. out.srt ..
                " via File > Import > Subtitle.")
        end
    else
        log("Import " .. out.srt .. " via File > Import > Subtitle.")
    end
end

-- ==== entry point ====

function M.run(cfg)
    local app = resolve
    if app == nil and fusion then
        local ok, r = pcall(function() return fusion:GetResolve() end)
        if ok then app = r end
    end
    if app == nil then
        log("Run this from DaVinci Resolve (Workspace > Scripts).")
        return
    end
    local ctx = {
        resolve = app,
        op = cfg.op,
        scope = cfg.scope or "current",
        track_index = cfg.track_index or 1,
        job_opts = cfg.job_opts or {},
        lib_dir = cfg.lib_dir,
        deps_dir = cfg.deps_dir,
        python = cfg.python,
        aspect = cfg.aspect or 0.5625,
        output_height = cfg.output_height or 1920,
        new_timeline_suffix = cfg.new_timeline_suffix,
        set_resolution = cfg.set_resolution,
    }
    if not collect(ctx) then return end
    local out = run_python(ctx)
    if out == nil then return end
    if ctx.op == "reframe" then
        apply_reframe(ctx, out)
    elseif ctx.op == "cut" then
        apply_cut(ctx, out)
    elseif ctx.op == "subtitles" then
        apply_subtitles(ctx, out)
    end
end

return M
