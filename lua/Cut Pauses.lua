-- Remove silent pauses from the clip under the playhead (new timeline).
-- Lua launcher: runs inside Resolve free 21.1+; heavy compute runs in Python outside Resolve.

local LIB_DIR = "__FACE_REFRAME_LIB__"
local DEPS_DIR = "__DEPS_DIR__"
local PYTHON = "__PYTHON__"

local lib = dofile(LIB_DIR .. "/fdr.lua")
lib.run({
    op = "cut",
    scope = "current",
    job_opts = {MINPAUSE = 0.35, PADDING = 0.16, MARGINDB = 8, RELEASEDB = 4, MINSPEECH = 0.15, FLOORPCT = 25},
    lib_dir = LIB_DIR,
    deps_dir = DEPS_DIR,
    python = PYTHON,
    new_timeline_suffix = " (no pauses)",
})
