-- Reframe the clip under the playhead for vertical output.
-- Lua launcher: runs inside Resolve free 21.1+; heavy compute runs in Python outside Resolve.

local LIB_DIR = "__FACE_REFRAME_LIB__"
local DEPS_DIR = "__DEPS_DIR__"
local PYTHON = "__PYTHON__"

local lib = dofile(LIB_DIR .. "/fdr.lua")
lib.run({
    op = "reframe",
    scope = "current",
    job_opts = {ASPECT = "9:16", OUTW = 1080, OUTH = 1920, KEYTOL = 0.5, ZOOM = 1.0, VERTICAL = "1"},
    lib_dir = LIB_DIR,
    deps_dir = DEPS_DIR,
    python = PYTHON,
    aspect = 0.5625,
    output_height = 1920,
    set_resolution = true,
})
