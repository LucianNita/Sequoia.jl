using Documenter
using Sequoia

makedocs(
    sitename = "Sequoia.jl",
    modules = [Sequoia],  
    format = Documenter.HTML(),
    pages = [
        "Introduction" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
        ]
)

deploydocs(
    repo = "github.com/LucianNita/Sequoia.jl.git",
    devbranch = "main"
)