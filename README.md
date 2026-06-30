# Ferrite-FEM-Sample-Notebook
A Pluto.jl notebook containing an example of a how you can use Ferrite.jl to solve a physical problem using the finite element menthod (FEM). This walkthrough follows the implementation of a heat equation solver coupled with a kinetic based metallic phase to hardness estimator.
## The code lives here

## How to run
1. Install the Julia programming language from julialang.org
2. Clone this repositiory (or alternatively download all the files).
3. Start Julia in the project folder and ensure the envirnoment is set; (run ``]activate .`` and ``]instantiate`` in the Julia REPL).
4. Run a Pluto.jl instance on your machine; (``using Pluto`` and then ``Pluto.run()`` in the Julia REPL).
5. A browser tab of Pluto should've automatically opened in the default browser.
6. From that page select ``laser_hardening_walkthrough.jl``. (Pluto will automatically manage dependencies from the imported file; consequently, the first execution will be slower as everything required auto-installs). 
