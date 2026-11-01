-- Transcribe every clip and import word-level subtitles into the timeline.
-- Lua launcher: runs inside Resolve free 21.1+; heavy compute runs in Python outside Resolve.

local LIB_DIR = "__FACE_REFRAME_LIB__"
local DEPS_DIR = "__DEPS_DIR__"
local PYTHON = "__PYTHON__"

local lib = dofile(LIB_DIR .. "/fdr.lua")
lib.run({
    op = "subtitles",
    scope = "all",
    job_opts = {MAXWORDS = 3, MAXDUR = 1.6},
    lib_dir = LIB_DIR,
    deps_dir = DEPS_DIR,
    python = PYTHON,
})
