using Documenter
using Sequoia

makedocs(
    sitename = "Sequoia.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/LucianNita/Sequoia.jl.git",
    devbranch = "main"
)