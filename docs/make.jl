using Documenter
using SimUtils

DocMeta.setdocmeta!(SimUtils, :DocTestSetup, :(using SimUtils); recursive=true)

const REPO_URL = "github.com/amirabbasi-physics/SimUtils.jl.git"
const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const GITHUB_REMOTE = Documenter.Remotes.GitHub("amirabbasi-physics", "SimUtils.jl")

makedocs(
    sitename = "SimUtils.jl",
    authors = "Amir Abbasi and contributors",
    modules = [SimUtils],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://amirabbasi-physics.github.io/SimUtils.jl",
        collapselevel = 1,
        edit_link = "main",
    ),
    remotes = Dict(ROOT_DIR => GITHUB_REMOTE),
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/amirabbasi-physics/SimUtils.jl.git",
    devbranch = "main",
)
