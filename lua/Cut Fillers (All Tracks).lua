-- Remove pauses AND filler words from every clip on every video track.
-- Lua launcher: runs inside Resolve free 21.1+; heavy compute runs in Python outside Resolve.

local LIB_DIR = "__FACE_REFRAME_LIB__"
local DEPS_DIR = "__DEPS_DIR__"
local PYTHON = "__PYTHON__"

local lib = dofile(LIB_DIR .. "/fdr.lua")
lib.run({
    op = "cut",
    scope = "all",
    job_opts = {MINPAUSE = 0.35, PADDING = 0.16, MARGINDB = 8, RELEASEDB = 4, MINSPEECH = 0.15, FLOORPCT = 25, FILLERS = "1"},
    lib_dir = LIB_DIR,
    deps_dir = DEPS_DIR,
    python = PYTHON,
    new_timeline_suffix = " (no pauses)",
})
