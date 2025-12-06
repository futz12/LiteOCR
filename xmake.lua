add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")

set_encodings("utf-8")

set_languages("c++20", "c11")

if is_plat("windows") then
    add_defines("NOMINMAX")
end

add_requires("ncnn master", {
    configs = {
        vulkan = true,
    }
})

add_requires("opencv-mobile")

add_includedirs("include/")
add_includedirs("src/")

target("LiteOCR")
    set_kind("static")
    add_files("src/backend/*.cpp")

    add_files("src/LiteOCREngine.cpp")

    add_packages("ncnn", "opencv-mobile")


function add_test(repo)
    target("test_" .. repo)
        set_kind("binary")
        add_includedirs("tests/")
        add_files("tests/test_" .. repo .. ".cpp")
        add_deps("LiteOCR")
        add_packages("ncnn", "opencv-mobile")

        set_rundir("$(projectdir)/")
end

add_test("detector")
add_test("recognizer")
add_test("textlineori")
add_test("baseocr")
add_test("docori")