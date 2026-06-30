### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 6683b50e-e01f-4b44-9490-f9c48083328f
begin
    using Ferrite                      # the FEM toolkit: meshes, shape functions, assembly
    using PlutoUI                      # sliders, table-of-contents
    import WGLMakie as Mke             # browser-native plotting (qualified to avoid name clashes)
    using SparseArrays
    using LinearAlgebra
    Mke.activate!()
end

# ╔═╡ f12c0002-0002-4a02-8b02-000000000002
# FerriteGmsh brings `gmsh` (CAD import + meshing) and `togrid` (Gmsh → Ferrite.Grid).
using FerriteGmsh

# ╔═╡ 4a47cd69-8735-4c8d-9079-11a9c18ebb76
md"""
# 🔥 From the heat equation to hardened steel

### An interactive walk-through of a laser-hardening FEM solver

This notebook builds — piece by piece — a finite-element solver that predicts the
**hardness** of a steel surface scanned by a moving laser. We follow three stages:

1. **Heat transfer** — solve for the temperature history `T(r, t)` as a laser sweeps the surface.
2. **Metallurgy** — at every point, turn that thermal cycle into **austenite** and **martensite** fractions.
3. **Hardness** — combine the phases into a hardness field `H = Σᵢ fᵢ Hᵢ`.

Everything runs in your browser through **WGLMakie**. Drag the sliders to explore.
Each section first *explains* a piece of physics, then shows the *code* that
discretises it, then *plots* the result — ending with a map of the hardened track.

> ⏱️ The one heavy cell is the time-marching solve (a few seconds on a coarse mesh);
> every plot afterwards just reads its stored output.
"""

# ╔═╡ 393c4a7b-57f8-4da9-83ad-bd7361f3504b
TableOfContents(title="📖 Contents", depth=2)

# ╔═╡ 24111f3d-b0f3-4b38-a257-bd26b2846831
md"""
## 1. The physics

The temperature field `T(r, t)` in the steel block obeys the **transient
heat-conduction equation**

```
ρ cₚ ∂T/∂t = ∇·(k ∇T) + q_v          (heat stored = conduction in + bulk source)
```

and every one of its six faces exchanges heat through a single **flux balance**

```
k (n·∇T) = q_s − h(T − T₀) − εσ(T⁴ − T₀⁴)
            └laser┘  └convection┘   └radiation┘
```

The laser heats only the **top surface** (`q_s`); all faces lose heat by convection
(`h`) and radiation (`εσ`). Discretising with FEM — multiply by a test function,
integrate by parts — turns the PDE into a **matrix ODE** for the nodal temperatures
`θ(t)`:

```
M θ̇ = f − K θ
│      │   │
mass  load stiffness (conduction + convection + radiation)
```

The rest of the notebook *builds these matrices term by term*, marches the ODE in
time, and then reads the hardness off the thermal history.

!!! note "A normalisation that trips people up"
	The code is divided through by `ρcₚ`, so **code `k` is the thermal *diffusivity***
	(not conductivity), lengths are in **mm**, and temperature is **absolute Kelvin**
	with the ambient pinned at `T₀ = 0 K` (so the `T⁴` radiation term is consistent).
"""

# ╔═╡ c9cf292c-291e-46f0-8661-170d885583ec
md"""
## 2. The mesh and the FEM scaffolding

We discretise a `100 × 50 × 15` block with **linear tetrahedra** (P1 elements).
Four objects recur in every FEM code:

- **`grid`** — the mesh (nodes + elements).
- **`CellValues` / `FacetValues`** — caches of the shape functions `uᵢ`, their
  gradients `∇uᵢ`, and the quadrature weight×Jacobian `dΩ` at each Gauss point.
  Element integrals are just sums over quadrature points of these.
- **`DofHandler`** — assigns each nodal unknown a global index (a *degree of freedom*).
- **`allocate_matrix(dh)`** — a sparse matrix pre-sized to the mesh connectivity.
"""

# ╔═╡ 1145cace-7d90-470a-8f94-1710f67de10a
begin
    # geometry of the steel block (mm):  [0,100] × [0,50] × [0,15]
    L, W, hz = 100.0, 50.0, 15.0
    el_ρ = 0.3                                    # element density  [elements / unit length]
    nels = round.(Int, (el_ρ*L, el_ρ*W, el_ρ*hz))
    grid = generate_grid(Tetrahedron, nels, Vec((0.,0.,0.)), Vec((L, W, hz)))

    ip  = Lagrange{RefTetrahedron, 1}()           # linear (P1) shape functions
    qr  = QuadratureRule{RefTetrahedron}(2)        # volume quadrature
    fqr = FacetQuadratureRule{RefTetrahedron}(3)   # surface quadrature
    cv  = CellValues(qr, ip)                       # uᵢ, ∇uᵢ, dΩ in the volume
    fv  = FacetValues(fqr, ip)                     # uᵢ, dΓ on the faces

    dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
    nd = ndofs(dh)

    Γtop = getfacetset(grid, "top")                # laser-irradiated face
    Γall = union(Γtop,                             # all six faces lose heat to ambient
                 getfacetset(grid, "bottom"),
                 getfacetset(grid, "left"),  getfacetset(grid, "right"),
                 getfacetset(grid, "front"), getfacetset(grid, "back"))
    (elements = getncells(grid), dofs = nd)
end

# ╔═╡ d14d70dc-819e-489f-a5b3-72f8c43318df
begin
    # Map each dof to its physical coordinate (dof and node numbering can differ),
    # then pick a surface probe near the start of the scan track for later plots.
    dof_coords = Vector{Vec{3,Float64}}(undef, nd)
    for cell in CellIterator(dh)
        cc = getcoordinates(cell)
        for (ld, gd) in enumerate(celldofs(cell))
            dof_coords[gd] = cc[ld]
        end
    end
    top_dofs = [i for i in 1:nd if dof_coords[i][3] > hz - 1e-6]
    xprobe, yprobe = 35.0, W/2
    probe_dof = top_dofs[argmin([(dof_coords[i][1]-xprobe)^2 +
                                 (dof_coords[i][2]-yprobe)^2 for i in top_dofs])]
    (probe_location = round.(Tuple(dof_coords[probe_dof]), digits=1),)
end

# ╔═╡ a4b51bd4-2ccf-4858-b1d4-ddbb5b9ba73e
md"The mesh nodes (coloured by height `z`):"

# ╔═╡ 99299eed-c933-4d0e-8f80-a7fde0c4cbe6
let
    xs = [p[1] for p in dof_coords]
    ys = [p[2] for p in dof_coords]
    zs = [p[3] for p in dof_coords]
    fig = Mke.Figure(size = (840, 360))
    ax  = Mke.Axis3(fig[1,1], aspect = :data,
                    title = "Tetrahedral mesh — $(getncells(grid)) elements, $nd nodes")
    Mke.scatter!(ax, xs, ys, zs, color = zs, colormap = :viridis, markersize = 5)
    fig
end

# ╔═╡ 3085b0eb-1b83-4943-84e9-c2c583c29b63
md"""
## 3. Assembling the mass and stiffness matrices

Two matrices are constant in time, so we build them **once**:

- **Mass** `M = ∫ uᵢ uⱼ dΩ` — couples the time derivative `θ̇`.
- **Volume stiffness** `K_v = ∫ k ∇uᵢ·∇uⱼ dΩ` — conduction.

The **assembly loop** is the heart of FEM: for each element we build a small dense
element matrix by summing over quadrature points, then *scatter* it into the global
sparse matrix with `assemble!`.
"""

# ╔═╡ 599d208d-5d1d-4277-af4c-f1f62fbe0809
begin
    # physical coefficients, in the code's ρcₚ-normalised units
    k_diff = 5.0       # thermal DIFFUSIVITY  k_phys/ρcₚ   [length²/time]
    h_conv = 0.001     # convection coefficient  h_phys/k_phys
    T0     = 0.0       # ambient / sink temperature [K]
    ε      = 0.7       # emissivity
    σ      = 1.42e-11  # Stefan–Boltzmann in code units
    εσ     = ε * σ
end

# ╔═╡ fbfc9465-5602-4e96-9a97-70b318f40960
# Assemble the volume mass M (∫uᵢuⱼ) and stiffness K_v (∫k∇uᵢ·∇uⱼ) in one pass.
function assemble_MK!(M, K, cv, dh, k)
    n = getnbasefunctions(cv); Me = zeros(n, n); Ke = zeros(n, n)
    aM = start_assemble(M); aK = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cv, cell); fill!(Me, 0); fill!(Ke, 0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for i in 1:n
                vi = shape_value(cv, qp, i); ∇vi = shape_gradient(cv, qp, i)
                for j in 1:n
                    Me[i, j] += vi * shape_value(cv, qp, j) * dΩ        # mass
                    Ke[i, j] += k * (∇vi ⋅ shape_gradient(cv, qp, j)) * dΩ  # stiffness
                end
            end
        end
        assemble!(aM, celldofs(cell), Me)
        assemble!(aK, celldofs(cell), Ke)
    end
    return M, K
end

# ╔═╡ 3aa2ace5-0ed0-447f-b342-7372026eeffa
# Robin (convective) SURFACE stiffness K_s = ∫ h·k uᵢuⱼ ds and constant load f_s.
# The factor h·k (not bare h) matches the analytical convention ∂ₙu = −h·u.
function assemble_robin!(Ks, fs, fv, dh, facets, k, hc, T0)
    n = getnbasefunctions(fv); Ke = zeros(n, n); fe = zeros(n)
    asm = start_assemble(Ks, fs)
    for fc in FacetIterator(dh, facets)
        reinit!(fv, fc); fill!(Ke, 0); fill!(fe, 0)
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            for i in 1:n
                vi = shape_value(fv, qp, i)
                fe[i] += hc*k*T0 * vi * dΓ
                for j in 1:n
                    Ke[i, j] += hc*k * vi * shape_value(fv, qp, j) * dΓ
                end
            end
        end
        assemble!(asm, celldofs(fc), Ke, fe)
    end
    return Ks, fs
end

# ╔═╡ a0581bd4-a18d-428f-959f-cbc048b38d94
begin
    M  = allocate_matrix(dh)
    Kv = allocate_matrix(dh)
    Ks = allocate_matrix(dh)
    fs = zeros(nd)
    assemble_MK!(M, Kv, cv, dh, k_diff)
    assemble_robin!(Ks, fs, fv, dh, Γall, k_diff, h_conv, T0)
    K = Kv + Ks                 # total conduction operator = volume + Robin surface
    sum(M)                      # ← should be 75000 (the box volume): an assembly sanity check
end

# ╔═╡ 97490518-3fc3-4812-bd5c-a150e690063b
md"""
The number printed above is `sum(M)` — it integrates the constant field `1` over the
domain, so it must equal the **box volume `100·50·15 = 75000`**. Matching it to
machine precision is the classic check that the mass matrix assembled correctly.

Below, the **sparsity pattern** of `M` and `K`: each black dot is a nonzero, i.e. a
pair of nodes that share an element. Almost everything is zero (distant nodes don't
interact), which is why both matrices are stored sparse.
"""

# ╔═╡ 3a3e9025-77f7-40bb-a571-36fb7b0d9c12
let
    fig = Mke.Figure(size = (760, 360))
    for (c, (A, ttl)) in enumerate(((M, "mass  M"), (K, "stiffness  K")))
        I, J, _ = findnz(A)
        ax = Mke.Axis(fig[1, c], yreversed = true, aspect = Mke.DataAspect(),
                      title = "$ttl  —  $(nnz(A)) nonzeros of $(nd)²",
                      xlabel = "column", ylabel = "row")
        Mke.scatter!(ax, J, I, markersize = 1.3, color = :black)
    end
    fig
end

# ╔═╡ 7f067a7e-e6a1-4dca-8527-a11209ddec85
md"""
## 4. The moving laser source

This is what makes it a *laser* model. A Gaussian spot of peak flux `P_laser` and
radius `r_spot` scans in `+x` at speed `v_scan` along the mid-width line, then
**switches off** once it reaches the track end `x_off` (so heat doesn't pile up
against the far edge). The laser contributes a **load vector**

```
q_s,ⱼ(t) = ∫_Γtop uⱼ q_laser(x, y, t) ds
```

rebuilt every step because the spot moves. Drag the slider to watch it travel.
"""

# ╔═╡ 343aaee6-ce7b-41b4-be32-e35d775c5bd0
begin
    P_laser = 3000.0   # peak surface flux (code units)
    r_spot  = 5.0      # 1/e spot radius
    v_scan  = 6.0      # scan speed
    x_start = 20.0     # spot x-position at t = 0
    x_off   = 90.0     # laser switches off when the spot reaches here
    y_scan  = W/2      # scan line (mid-width)
    x_spot(t) = x_start + v_scan * t
    q_laser(x, y, t) = x_spot(t) > x_off ? 0.0 :
        P_laser * exp(-((x - x_spot(t))^2 + (y - y_scan)^2) / (2 * r_spot^2))
end

# ╔═╡ fb808342-378a-49b6-aa72-bc73010f4c1c
md"""
**Scan time** `t =` $(@bind t_laser PlutoUI.Slider(0.0:0.5:16.0, default=4.0, show_value=true)) `s`
"""

# ╔═╡ b54ac871-c233-4976-99db-34f3033dc364
let
    xs = range(0, L, 200); ys = range(0, W, 100)
    Q = [q_laser(x, y, t_laser) for x in xs, y in ys]
    fig = Mke.Figure(size = (840, 300))
    ax = Mke.Axis(fig[1,1], aspect = Mke.DataAspect(),
                  title = "Laser surface flux  q_s(x, y, t = $(t_laser) s)",
                  xlabel = "x [mm]", ylabel = "y [mm]")
    hm = Mke.heatmap!(ax, xs, ys, Q, colormap = :inferno, colorrange = (0, P_laser))
    Mke.lines!(ax, [0, L], [y_scan, y_scan], color = (:white, 0.5), linestyle = :dash)
    Mke.Colorbar(fig[1,2], hm, label = "q_s")
    fig
end

# ╔═╡ 472c663c-e762-4d3d-9bc0-2541b641ec55
# Moving-laser load vector  q_s,ⱼ(t) = ∫_Γtop uⱼ q_laser ds  (rebuilt every step).
# `spatial_coordinate` recovers the physical (x,y) of each quad point so the moving
# Gaussian can be sampled there.
function assemble_qs!(f, t, fv, dh, facets)
    n = getnbasefunctions(fv); fe = zeros(n); fill!(f, 0)
    for fc in FacetIterator(dh, facets)
        reinit!(fv, fc); fill!(fe, 0); coords = getcoordinates(fc)
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            x  = spatial_coordinate(fv, qp, coords)
            qv = q_laser(x[1], x[2], t)
            for i in 1:n
                fe[i] += qv * shape_value(fv, qp, i) * dΓ
            end
        end
        assemble!(f, celldofs(fc), fe)
    end
    return f
end

# ╔═╡ 97188e99-a10b-4955-ad79-f1be3145a71a
md"""
## 5. Radiation, linearised

Radiation `εσ(T⁴ − T₀⁴)` is the only **nonlinear** term. The trick (used in the
production solver) is a *frozen-coefficient* linearisation: within a step,
approximate `T⁴(tₙ) ≈ T³(tₙ₋₁)·T(tₙ)`, taking the cube of the *previous* step's
temperature as a known coefficient. The radiative surface integral then becomes a
**matrix** acting linearly on the current `T`:

```
R_ⱼᵢ = ∫_∂Ω uⱼ · T³ · uᵢ ds      ("the T³ matrix")
```

assembled only over the boundary facets and refreshed once per step. The stiffness
becomes `K + εσ·R`. (Here convection/radiation are deliberately weak sinks — the
laser dominates — but the machinery is the same as the full solver's.)
"""

# ╔═╡ 8dc6a16d-3f94-444c-9670-0c10cd91b475
# The "T³ matrix"  R_ⱼᵢ = ∫_∂Ω uⱼ T³ uᵢ ds  on the radiating facets. `u` is the
# current nodal field; `function_value` interpolates T at each quad point and we cube it.
function assemble_radiation!(R, u, fv, dh, facets)
    n = getnbasefunctions(fv); Re = zeros(n, n); ue = zeros(n)
    asm = start_assemble(R)
    for fc in FacetIterator(dh, facets)
        reinit!(fv, fc); fill!(Re, 0); ue .= @view u[celldofs(fc)]
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            T³ = function_value(fv, qp, ue)^3
            for i in 1:n
                vi = shape_value(fv, qp, i)
                for j in 1:n
                    Re[i, j] += T³ * vi * shape_value(fv, qp, j) * dΓ
                end
            end
        end
        assemble!(asm, celldofs(fc), Re)
    end
    return R
end

# ╔═╡ 82d3a719-29ed-4611-b9f8-6d378474453a
md"""
## 6. Metallurgy: temperature → phases → hardness

With a temperature history in hand, the metallurgy is what turns heat into hardness.
Two transformations and a mixing rule:

**Austenitisation on heating — JMAK.** Above `Ac1`, steel transforms to **austenite**
by a diffusional, *time-dependent* process (Johnson–Mehl–Avrami–Kolmogorov), with an
Arrhenius rate `k(T) = k₀·exp(−Q/RT)` that is fast when hot:

```
dX_a/dt = n·k(T)·(1 − X_a)·[−ln(1 − X_a)]^((n−1)/n)
```

**Martensite on cooling — Koistinen–Marburger.** When the austenitised metal then
self-quenches (conduction into the cold bulk), austenite transforms to hard
**martensite**, an *athermal* process depending only on how far below `Ms` you are:

```
f_KM(T) = 1 − exp(−α(Ms − T)),    T < Ms
```

**Phase bookkeeping (the subtle part).** `X_a` and `X_m` are *cumulative* variables —
they do **not** sum to 1. Martensite forms *out of* austenite, so `X_m ≤ X_a ≤ 1`.
The three *true* fractions that sum to 1, and that feed the hardness, are:

| phase | fraction | hardness Hᵢ [HV] |
|---|---|---|
| base ferrite/pearlite | `1 − X_a` | 220 |
| retained austenite | `X_a − X_m` | 300 |
| martensite | `X_m` | 750 |

and the **rule of mixtures** gives `H = (1−X_a)·H_fp + (X_a−X_m)·H_aust + X_m·H_mart`.
"""

# ╔═╡ e54429c6-f52b-4e69-bef1-ebe8a180df99
begin
    Ac1 = 1000.0      # austenite-start   [K]
    Ac3 = 1150.0      # austenite-finish  [K]
    Ms  = 600.0       # martensite-start  [K]
    α_KM = 0.011      # Koistinen–Marburger constant [1/K]
    n_av = 2.5        # Avrami exponent
    Q_a  = 2.5e5      # activation energy [J/mol]
    Rg   = 8.314      # gas constant
    k0_a = 1.0e12     # pre-exponential   [1/s]
    X_seed = 1.0e-4   # JMAK incubation nucleus (bootstraps the singular rate form)
    H_fp   = 220.0    # ferrite/pearlite  [HV]
    H_aust = 300.0    # retained austenite [HV]
    H_mart = 750.0    # martensite        [HV]
end

# ╔═╡ dd5ffd5f-f3ab-4c18-a17b-c8542c14d6ec
begin
    k_jmak(T) = T > Ac1 ? k0_a * exp(-Q_a / (Rg * T)) : 0.0

    # one forward-Euler step of the JMAK austenite fraction
    function jmak_step(Xa, T, dt)
        (T ≤ Ac1 || Xa ≥ 1.0) && return Xa
        Xe   = clamp(max(Xa, X_seed), X_seed, 1 - 1e-9)
        rate = n_av * k_jmak(T) * (1 - Xe) * (-log(1 - Xe))^((n_av - 1) / n_av)
        return min(1.0, Xa + rate * dt)
    end

    # Koistinen–Marburger martensite fraction of the available austenite (athermal)
    km_fraction(T) = T < Ms ? (1 - exp(-α_KM * (Ms - T))) : 0.0

    # weighted hardness from the three true phase fractions
    phase_hardness(xa, xm) = (1 - xa) * H_fp + (xa - xm) * H_aust + xm * H_mart
end

# ╔═╡ 3e705579-dc95-4dcb-8b4d-ee6c53bd5467
md"""
The kinetic laws, visualised: austenite **grows faster the hotter you are** (JMAK,
left); martensite **grows the colder you get below `Ms`** (KM, middle); and the
Arrhenius rate spans orders of magnitude over the austenitising range (right).
"""

# ╔═╡ bb37cb91-3f9c-47f7-866a-cd09d4e677d2
# helper: integrate the JMAK fraction along a constant-temperature hold, for plotting
function jmak_curve(T, ts)
    Xa = 0.0; out = zeros(length(ts))
    for k in 2:length(ts)
        Xa = jmak_step(Xa, T, ts[k] - ts[k-1]); out[k] = Xa
    end
    return out
end

# ╔═╡ 04b12484-2ab4-4a4a-adce-e12bfe6dc4f3
let
    fig = Mke.Figure(size = (980, 300))

    ax1 = Mke.Axis(fig[1,1], title = "JMAK — austenite vs time",
                   xlabel = "t [s]", ylabel = "X_a")
    ts = collect(range(0, 1.2, 240))
    for T in (1050.0, 1100.0, 1200.0, 1350.0)
        Mke.lines!(ax1, ts, jmak_curve(T, ts), label = "$(Int(T)) K")
    end
    Mke.axislegend(ax1, position = :rb)

    ax2 = Mke.Axis(fig[1,2], title = "Koistinen–Marburger — martensite vs T",
                   xlabel = "T [K]", ylabel = "fraction")
    Ts = collect(range(300, 650, 240))
    Mke.lines!(ax2, Ts, km_fraction.(Ts), color = :navy)
    Mke.vlines!(ax2, [Ms], color = (:gray, 0.6), linestyle = :dash)
    Mke.text!(ax2, Ms, 0.05, text = " Ms", color = :gray, align = (:left, :bottom))

    ax3 = Mke.Axis(fig[1,3], title = "Arrhenius rate k(T)", yscale = log10,
                   xlabel = "T [K]", ylabel = "k [1/s]")
    Tr = collect(range(1001, 1500, 240))
    Mke.lines!(ax3, Tr, k_jmak.(Tr), color = :firebrick)

    fig
end

# ╔═╡ d5bdf07f-0571-43b5-a0d6-d72ced82f9ba
md"""
## 7. Time integration — marching the coupled solve

We integrate `M θ̇ = f − (K + εσR) θ` with a transparent **backward-Euler** step.
Because the laser load `q_s` and the `T³` matrix `R` are *frozen* within a step, the
update is a single **linear solve**:

```
(M/Δt + K + εσR) uⁿ⁺¹ = (M/Δt) uⁿ + f_s + q_s(tₙ)
```

After each temperature step we **operator-split** the metallurgy: advance every node's
`X_a` / `X_m` explicitly along the just-computed temperature segment. The phase update
is purely local (no spatial coupling), so it is a simple per-node loop.

!!! tip "What the production solver does differently"
	The real solver replaces backward Euler with an adaptive **stiff Rosenbrock**
	integrator (DifferentialEquations.jl `Rodas5P`), runs the metallurgy in a
	callback that fires only on *accepted* steps, and threads the assembly + linear
	solve across all CPU cores. The physics is identical; this notebook keeps the
	stepper explicit so you can read it.
"""

# ╔═╡ ae1d4676-8228-4cb0-b546-c7711646fcaf
# Advance every node's phase state over one heat step (tₚ, t]. Temperature is taken
# piecewise-linearly between the endpoints; the per-node peak is exact as max(endpoints).
function advance_phases!(Xa, Xm, Tmax, uprev, u, Δt, nd; dtmax = 0.02)
    nsub = max(1, ceil(Int, Δt / dtmax)); hsub = Δt / nsub
    @inbounds for i in 1:nd
        up = uprev[i]; ui = u[i]
        Tmax[i] = max(Tmax[i], up, ui)
        for s in 1:nsub
            Ti = up + ((s - 0.5) / nsub) * (ui - up)
            if Ti > Ac1
                Xm[i] = 0.0                          # re-austenitised: martensite reverts
                Xa[i] = jmak_step(Xa[i], Ti, hsub)   # grow austenite (JMAK)
            else
                xmeq = Xa[i] * km_fraction(Ti)       # KM martensite of available austenite
                xmeq > Xm[i] && (Xm[i] = xmeq)
            end
        end
    end
    return nothing
end

# ╔═╡ 7eb91bfa-3737-4b33-abbb-72646a232ccb
# The full operator-split solve. Returns the final phase fields, a probe time series,
# and a handful of temperature snapshots for the scrubber below.
function solve_hardening(M, K, fs, dh, fv, Γtop, Γall, nd; Δt = 0.1, t_final = 16.0, snap_dt = 0.5)
    R = allocate_matrix(dh)
    q = zeros(nd)
    u = zeros(nd)                                    # T₀ = 0 everywhere
    Xa = zeros(nd); Xm = zeros(nd); Tmax = fill(-Inf, nd)

    t_snaps = Float64[0.0]; T_snaps = [copy(u)]; next_snap = snap_dt
    pt = Float64[]; pT = Float64[]; pXa = Float64[]; pXm = Float64[]

    assemble_qs!(q, 0.0, fv, dh, Γtop)
    assemble_radiation!(R, u, fv, dh, Γall)

    nsteps = round(Int, t_final / Δt); t = 0.0
    for _ in 1:nsteps
        uprev = copy(u); tnew = t + Δt
        assemble_qs!(q, tnew, fv, dh, Γtop)          # move the laser load to tₙ
        assemble_radiation!(R, u, fv, dh, Γall)      # freeze T³ at the current state
        A = K + εσ * R
        u = (M ./ Δt .+ A) \ ((M * u) ./ Δt .+ fs .+ q)   # backward-Euler linear solve
        advance_phases!(Xa, Xm, Tmax, uprev, u, Δt, nd)
        t = tnew

        push!(pt, t); push!(pT, u[probe_dof])
        push!(pXa, Xa[probe_dof]); push!(pXm, Xm[probe_dof])
        if t >= next_snap - 1e-9
            push!(t_snaps, t); push!(T_snaps, copy(u)); next_snap += snap_dt
        end
    end

    HV = phase_hardness.(Xa, Xm)
    return (; t_snaps, T_snaps, Xa, Xm, Tmax, HV,
              probe_t = pt, probe_T = pT, probe_Xa = pXa, probe_Xm = pXm)
end

# ╔═╡ 58e8c1cb-3b83-4233-b4f1-dc74ac95e5ef
# ⏱️ THE HEAVY CELL — runs the whole time-marching solve once.
result = solve_hardening(M, K, fs, dh, fv, Γtop, Γall, nd; Δt = 0.1, t_final = 16.0)

# ╔═╡ e4abc8a4-0d67-4c13-9890-25ce9a3f8411
md"""
### Results at a glance
"""

# ╔═╡ 0f5681d2-dc94-431c-acf9-022a6e753dc5
let
    vf(mask) = round(100 * count(mask) / nd, digits = 2)
    Markdown.parse("""
    | quantity | value |
    |:--|:--|
    | peak temperature reached | **$(round(maximum(result.Tmax), digits=1)) K** |
    | volume austenitised (Xₐ > 0.5) | $(vf(result.Xa .> 0.5)) % |
    | volume martensitic (Xₘ > 0.5) | $(vf(result.Xm .> 0.5)) % |
    | peak austenite / martensite | $(round(maximum(result.Xa), digits=3)) / $(round(maximum(result.Xm), digits=3)) |
    | final hardness range | $(round(minimum(result.HV), digits=1)) – $(round(maximum(result.HV), digits=1)) HV |
    """)
end

# ╔═╡ d3d55413-dee9-470e-aff5-28889fcbf850
md"""
## 8. The hardened material

We read the fields off the mesh with Ferrite's `PointEvalHandler`, which evaluates the
FE interpolation at any set of points — giving clean gridded **heatmaps** of the top
surface and a cross-section, instead of raw scatter.
"""

# ╔═╡ d459110a-880c-4e4b-b691-be1de65bee3f
# Evaluate a nodal field on a regular grid just under the TOP surface → heatmap data.
function surface_field(values; nx = 150, ny = 75)
    xs = range(0.7, L - 0.7, nx); ys = range(0.7, W - 0.7, ny)
    pts = vec([Vec(x, y, hz - 1e-3) for x in xs, y in ys])
    ph  = PointEvalHandler(grid, pts)
    vals = evaluate_at_points(ph, dh, values, :u)
    g = [v === missing ? NaN : float(v) for v in vals]
    return xs, ys, reshape(g, nx, ny)
end

# ╔═╡ 427c01f3-b39e-4ec7-b986-4140944a7f68
# Evaluate a nodal field on the x–z mid-plane (y = W/2) → cross-section heatmap data.
function section_field(values; nx = 150, nz = 45, y = W / 2)
    xs = range(0.7, L - 0.7, nx); zs = range(0.1, hz - 0.05, nz)
    pts = vec([Vec(x, y, z) for x in xs, z in zs])
    ph  = PointEvalHandler(grid, pts)
    vals = evaluate_at_points(ph, dh, values, :u)
    g = [v === missing ? NaN : float(v) for v in vals]
    return xs, zs, reshape(g, nx, nz)
end

# ╔═╡ 18d6d29e-8141-4595-a819-155691b9eaae
md"""
The **peak temperature envelope** (top) drives everything: where it crossed `Ac1` the
metal austenitised (`X_a`, middle), and where that zone then quenched below `Ms` it
became martensite (`X_m`, bottom). The bright stripe is the laser's scan track.
"""

# ╔═╡ 8ae211dd-b10e-40a3-8964-725db9f8e04d
let
    fig = Mke.Figure(size = (1000, 760))
    function panel!(row, vals, ttl, crange, cmap, lbl)
        xs, ys, F = surface_field(vals)
        ax = Mke.Axis(fig[row, 1], aspect = Mke.DataAspect(), title = ttl,
                      xlabel = "x [mm]", ylabel = "y [mm]")
        hm = Mke.heatmap!(ax, xs, ys, F, colormap = cmap, colorrange = crange)
        Mke.Colorbar(fig[row, 2], hm, label = lbl)
    end
    panel!(1, result.Tmax, "Peak temperature reached  (top surface)", (0, maximum(result.Tmax)), :hot, "T_peak [K]")
    panel!(2, result.Xa,   "Austenite fraction  X_a  (final)",        (0, 1), :viridis, "X_a")
    panel!(3, result.Xm,   "Martensite fraction  X_m  (final)",       (0, 1), :viridis, "X_m")
    fig
end

# ╔═╡ b8f0a112-97a1-4cb5-9c31-8f03255ef815
md"""
### The hardness field — the deliverable

Top: hardness over the surface (the hardened track in bright `750 HV` martensite
against the soft `220 HV` base metal). Bottom: a cross-section showing the **case
depth** — hardening is concentrated in a shallow layer under the scanned line.
"""

# ╔═╡ 62f210b4-5ab3-4a9d-bd92-42dd89c45051
let
    fig = Mke.Figure(size = (1000, 580))

    xs, ys, H = surface_field(result.HV)
    ax1 = Mke.Axis(fig[1,1], aspect = Mke.DataAspect(),
                   title = "Hardness H  (top surface) — the hardened track",
                   xlabel = "x [mm]", ylabel = "y [mm]")
    hm1 = Mke.heatmap!(ax1, xs, ys, H, colormap = :plasma, colorrange = (220, 750))
    Mke.Colorbar(fig[1,2], hm1, label = "HV")

    xs2, zs2, H2 = section_field(result.HV)
    ax2 = Mke.Axis(fig[2,1], aspect = Mke.DataAspect(),
                   title = "Hardness cross-section at y = W/2 — case depth",
                   xlabel = "x [mm]", ylabel = "z [mm]")
    hm2 = Mke.heatmap!(ax2, xs2, zs2, H2, colormap = :plasma, colorrange = (220, 750))
    Mke.Colorbar(fig[2,2], hm2, label = "HV")

    fig
end

# ╔═╡ 5f087f57-147c-4743-a119-866fef96baef
md"""
### Watch the surface temperature evolve

$(@bind tsnap_idx PlutoUI.Slider(1:length(result.t_snaps), default=min(9, length(result.t_snaps)), show_value=false))
"""

# ╔═╡ 4da1ec68-0f85-4b01-9bb6-ac521daf9884
let
    tmax = maximum(maximum, result.T_snaps)
    xs, ys, F = surface_field(result.T_snaps[tsnap_idx])
    fig = Mke.Figure(size = (840, 320))
    ax = Mke.Axis(fig[1,1], aspect = Mke.DataAspect(),
                  title = "Surface temperature at  t = $(round(result.t_snaps[tsnap_idx], digits=2)) s",
                  xlabel = "x [mm]", ylabel = "y [mm]")
    hm = Mke.heatmap!(ax, xs, ys, F, colormap = :hot, colorrange = (0, tmax))
    Mke.Colorbar(fig[1,2], hm, label = "T [K]")
    fig
end

# ╔═╡ a204af3d-1bf6-459c-9d5b-ad7bea1cba15
md"""
### The thermal cycle at one point

At the probe (near the start of the track) you can see the whole story in one plot:
temperature spikes as the laser passes, austenite (`X_a`) forms once `T` exceeds
`Ac1`, then martensite (`X_m`) forms as the point self-quenches below `Ms`. Both end
near 1 — the signature of a fully hardened point.
"""

# ╔═╡ 2c0c84ad-245f-40dd-be88-0299f1f2cc51
let
    Tpk = maximum(result.Tmax)
    fig = Mke.Figure(size = (900, 360))
    ax = Mke.Axis(fig[1,1], xlabel = "t [s]", ylabel = "fraction   /   T·T_peak⁻¹",
                  title = "Thermal cycle & phase fractions at the probe (x ≈ $(round(dof_coords[probe_dof][1], digits=1)) mm, surface)")
    Mke.lines!(ax, result.probe_t, result.probe_T ./ Tpk, color = :firebrick, label = "T / T_peak")
    Mke.lines!(ax, result.probe_t, result.probe_Xa, color = :seagreen, linewidth = 2, label = "X_a (austenite)")
    Mke.lines!(ax, result.probe_t, result.probe_Xm, color = :navy, linewidth = 2, label = "X_m (martensite)")
    Mke.hlines!(ax, [Ac1/Tpk, Ms/Tpk], color = (:gray, 0.6), linestyle = :dash)
    Mke.text!(ax, result.probe_t[end], Ac1/Tpk, text = " Ac1", align = (:right, :bottom), color = :gray)
    Mke.text!(ax, result.probe_t[end], Ms/Tpk,  text = " Ms",  align = (:right, :bottom), color = :gray)
    Mke.axislegend(ax, position = :rc)
    fig
end

# ╔═╡ 2fef4b4e-6845-45ab-a56a-5c2d9ef3cf2a
md"""
### Case-depth profile

Following hardness straight down from the probe: a hard martensitic case near the
surface decaying to the soft core — exactly what laser surface hardening is for.
"""

# ╔═╡ a7e0ae6e-7085-482a-867c-e5f68f413c5e
let
    col = filter(i -> abs(dof_coords[i][1] - dof_coords[probe_dof][1]) < 1e-6 &&
                      abs(dof_coords[i][2] - dof_coords[probe_dof][2]) < 1e-6, 1:nd)
    ord  = sortperm([dof_coords[i][3] for i in col])
    z    = [dof_coords[i][3] for i in col[ord]]
    Hc   = result.HV[col[ord]]
    fig = Mke.Figure(size = (460, 440))
    ax = Mke.Axis(fig[1,1], xlabel = "hardness [HV]", ylabel = "z, height from bottom [mm]",
                  title = "Case-depth profile under the probe")
    Mke.lines!(ax, Hc, z, color = :purple, linewidth = 2)
    Mke.scatter!(ax, Hc, z, color = :purple, markersize = 8)
    fig
end

# ╔═╡ f12c0001-0001-4a01-8b01-000000000001
md"""
## 9. Arbitrary geometry — meshing a real part & projecting the laser

The solver so far lives on a tidy `100 × 50 × 15` box. Real workpieces are not
boxes — they are CAD parts with pockets, fillets and curved tops. This last section
keeps **all** of the physics and code above and changes only the *geometry*: it
imports a real CAD solid, meshes it, and teaches the laser to follow a surface that
is no longer flat, or even convex.

Three new ideas:

1. **Meshing a CAD solid.** A STEP/IGES file stores the exact surfaces bounding a
   solid, not a mesh. We hand it to Gmsh's CAD kernel, which fills the volume with
   tetrahedra, and convert the result to a Ferrite grid.
2. **Boundary from topology.** A CAD import has no named `top`/`bottom` faces — just
   one anonymous solid. We recover the exterior straight from the mesh connectivity,
   and pick out the **up-facing** facets the beam can actually strike.
3. **Projecting the laser onto the outermost face.** The spot is still a footprint in
   `(x, y)`; we deposit it on whichever up-facing facet lies beneath it — at whatever
   height `z`. Over a milled groove, the beam follows the surface *down*.

> ⏱️ This section meshes a part and runs a second (coarse) solve — another few
> seconds. On first run Pluto also installs **FerriteGmsh**, the Gmsh ↔ Ferrite bridge.
"""

# ╔═╡ f12c0003-0003-4a03-8b03-000000000003
# Import a CAD solid (STEP/IGES) through Gmsh's OpenCASCADE kernel, mesh the volume
# with tetrahedra (element size scaled to the part), and hand it back as a Ferrite grid.
function load_cad_grid(file; nominal_elems = 22)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        gmsh.model.occ.importShapes(replace(abspath(file), '\\' => '/'))  # read the B-rep
        gmsh.model.occ.synchronize()
        bb   = gmsh.model.getBoundingBox(-1, -1)          # (xmin,ymin,zmin, xmax,ymax,zmax)
        diag = hypot(bb[4]-bb[1], bb[5]-bb[2], bb[6]-bb[3])
        lc   = diag / nominal_elems                       # element size from the part scale
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc*0.4)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        gmsh.model.mesh.generate(3)                       # tetrahedralise the volume
        return togrid(), bb                               # current Gmsh model → Ferrite.Grid
    finally
        gmsh.finalize()
    end
end

# ╔═╡ f12c0004-0004-4a04-8b04-000000000004
# Load the test part — a block with a half-cylinder groove milled out of its top.
cad_grid, cad_bb = let
    f = joinpath(@__DIR__, "3d_objects", "mold.stp")
    isfile(f) || error("CAD part not found at $f — put mold.stp in ./3d_objects")
    load_cad_grid(f; nominal_elems = 22)
end

# ╔═╡ f12c0005-0005-4a05-8b05-000000000005
md"""
### Finding the surface from the mesh, not from names

`generate_grid` handed us named faces (`getfacetset(grid, "top")`). A CAD import
gives one anonymous solid, so we read the boundary off the **topology**: a facet is
on the exterior exactly when no second cell sits on its far side.

```
exterior  Γ_N      = every boundary facet        → convection + radiation
up-facing Γ_laser  = boundary facets with n̂·ẑ > 0 → the surface the laser can strike
```

`getnormal(fv, qp)` is the outward unit normal at a surface quadrature point;
averaging its `z`-component classifies each facet as up-facing, side-wall, or
down-facing.
"""

# ╔═╡ f12c0006-0006-4a06-8b06-000000000006
begin
    cad_dh = DofHandler(cad_grid); add!(cad_dh, :u, ip); close!(cad_dh)
    cad_nd = ndofs(cad_dh)

    # every dof's physical coordinate (for the surface scatter plots below)
    cad_dof_coords = Vector{Vec{3,Float64}}(undef, cad_nd)
    for cell in CellIterator(cad_dh)
        cc = getcoordinates(cell)
        for (ld, gd) in enumerate(celldofs(cell)); cad_dof_coords[gd] = cc[ld]; end
    end

    # the ENTIRE exterior boundary, straight from the mesh connectivity
    cad_topo = Ferrite.ExclusiveTopology(cad_grid)
    cad_ΓN   = Ferrite.create_boundaryfacetset(cad_grid, cad_topo, x -> true)

    # split off the UP-facing facets (n̂·ẑ > 0) and record each centroid, so we can
    # *see* the non-convex top (the groove dips below the surrounding flat surface)
    cad_Γlaser = Ferrite.OrderedSet{FacetIndex}()   # OrderedSet ⇒ works with FacetIterator
    lcx = Float64[]; lcy = Float64[]; lcz = Float64[]
    let fc = FacetCache(cad_dh)
        for fi in cad_ΓN
            reinit!(fc, fi); reinit!(fv, fc)
            nq = getnquadpoints(fv)
            nz = sum(getnormal(fv, qp)[3] for qp in 1:nq) / nq
            if nz > 0.05
                push!(cad_Γlaser, fi)
                cds = getcoordinates(fc)
                c = sum(spatial_coordinate(fv, qp, cds) for qp in 1:nq) / nq
                push!(lcx, c[1]); push!(lcy, c[2]); push!(lcz, c[3])
            end
        end
    end
    (cells = getncells(cad_grid), dofs = cad_nd,
     exterior_facets = length(cad_ΓN), laser_facing = length(cad_Γlaser))
end

# ╔═╡ f12c0007-0007-4a07-8b07-000000000007
let
    ymid = 0.5 * (cad_bb[2] + cad_bb[5])
    band = 0.10 * (cad_bb[5] - cad_bb[2])
    sel  = findall(i -> abs(lcy[i] - ymid) < band, eachindex(lcx))
    fig = Mke.Figure(size = (1000, 420))
    ax1 = Mke.Axis3(fig[1,1], aspect = :data, azimuth = 0.7π, elevation = 0.25π,
                    title = "Laser-facing surface, coloured by height z")
    sc = Mke.scatter!(ax1, lcx, lcy, lcz, color = lcz, colormap = :viridis, markersize = 7)
    Mke.Colorbar(fig[1,2], sc, label = "z [mm]")
    ax2 = Mke.Axis(fig[1,3], title = "Side view along the scan line (y ≈ mid)",
                   xlabel = "x [mm]", ylabel = "surface height z [mm]")
    Mke.scatter!(ax2, lcx[sel], lcz[sel], color = lcz[sel], colormap = :viridis, markersize = 7)
    fig
end

# ╔═╡ f12c0008-0008-4a08-8b08-000000000008
md"""
### Projecting the beam onto the outermost face

The moving spot is defined **only** in `(x, y)`. On the flat box top that footprint
sat at one height; here the up-facing surface wanders in `z`, and where the groove is
milled out the surface directly under the beam is *lower* than the surrounding top.

The projection falls out for free: we assemble the laser load over `Γ_laser`, and at
each facet quadrature point sample the footprint at that point's `(x, y)` — **whatever
its `z`**. One physical refinement: a vertical beam hitting a *tilted* facet spreads
its power over a larger area, so the absorbed flux carries the geometric factor
`n̂·ẑ` (the cosine of the tilt). On a flat top `n̂·ẑ = 1` and this reduces exactly to
the box model — and integrated over the patch it conserves the beam's total power.
"""

# ╔═╡ f12c0009-0009-4a09-8b09-000000000009
# Moving-laser load, PROJECTED onto the outermost face. Identical to `assemble_qs!`
# except the footprint is taken from a supplied `qfun(x,y,t)` and weighted by the
# tilt factor n̂·ẑ = getnormal(fv,qp)[3] (the vertical-beam-onto-tilted-facet cosine).
function assemble_qs_proj!(f, t, fv, dh, facets, qfun)
    n = getnbasefunctions(fv); fe = zeros(n); fill!(f, 0)
    for fc in FacetIterator(dh, facets)
        reinit!(fv, fc); fill!(fe, 0); coords = getcoordinates(fc)
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            x  = spatial_coordinate(fv, qp, coords)
            nz = getnormal(fv, qp)[3]                 # cosine projection factor
            qv = qfun(x[1], x[2], t) * nz
            for i in 1:n
                fe[i] += qv * shape_value(fv, qp, i) * dΓ
            end
        end
        assemble!(f, celldofs(fc), fe)
    end
    return f
end

# ╔═╡ f12c000a-000a-4a0a-8b0a-00000000000a
begin
    # a straight scan along the part's longest axis (x), through its centre in y, so
    # the beam crosses the groove. Spot size & path are scaled to the part's bbox.
    cad_Lx     = cad_bb[4] - cad_bb[1]
    cad_xlo    = cad_bb[1] + 0.12 * cad_Lx
    cad_xhi    = cad_bb[4] - 0.12 * cad_Lx
    cad_yscan  = 0.5 * (cad_bb[2] + cad_bb[5])
    cad_tscan  = 12.0
    cad_vscan  = (cad_xhi - cad_xlo) / cad_tscan
    cad_rspot  = 0.06 * cad_Lx
    cad_Plaser = 3000.0
    cad_xspot(t) = cad_xlo + cad_vscan * t
    q_laser_cad(x, y, t) = cad_xspot(t) > cad_xhi ? 0.0 :
        cad_Plaser * exp(-((x - cad_xspot(t))^2 + (y - cad_yscan)^2) / (2 * cad_rspot^2))
end

# ╔═╡ f12c000b-000b-4a0b-8b0b-00000000000b
begin
    # the SAME assembly routines from §3 — they take the grid/dh as arguments, so they
    # are geometry-agnostic and need no changes for the CAD mesh.
    M_cad  = allocate_matrix(cad_dh)
    Kv_cad = allocate_matrix(cad_dh)
    Ks_cad = allocate_matrix(cad_dh)
    fs_cad = zeros(cad_nd)
    assemble_MK!(M_cad, Kv_cad, cv, cad_dh, k_diff)
    assemble_robin!(Ks_cad, fs_cad, fv, cad_dh, cad_ΓN, k_diff, h_conv, T0)
    K_cad = Kv_cad + Ks_cad
    (mass_integral = sum(M_cad),)        # ≈ the part's volume — assembly sanity check
end

# ╔═╡ f12c000c-000c-4a0c-8b0c-00000000000c
md"""
### Run it on the part

The very same operator-split march from §7 — backward-Euler heat step, projected
laser load, frozen-`T³` radiation, then an explicit per-node phase update — now on the
CAD mesh. Only the laser assembly changed (it projects onto `Γ_laser`); everything
else is reused verbatim.
"""

# ╔═╡ f12c000d-000d-4a0d-8b0d-00000000000d
# ⏱️ THE SECOND HEAVY CELL — the hardening solve on the real part.
cad_result = let
    R = allocate_matrix(cad_dh); q = zeros(cad_nd); u = zeros(cad_nd)
    Xa = zeros(cad_nd); Xm = zeros(cad_nd); Tmax = fill(-Inf, cad_nd)
    Δt = 0.1; t_final = cad_tscan + 5.0
    assemble_qs_proj!(q, 0.0, fv, cad_dh, cad_Γlaser, q_laser_cad)
    assemble_radiation!(R, u, fv, cad_dh, cad_ΓN)
    nsteps = round(Int, t_final / Δt); t = 0.0
    for _ in 1:nsteps
        uprev = copy(u); tnew = t + Δt
        assemble_qs_proj!(q, tnew, fv, cad_dh, cad_Γlaser, q_laser_cad)
        assemble_radiation!(R, u, fv, cad_dh, cad_ΓN)
        A = K_cad + εσ * R
        u = (M_cad ./ Δt .+ A) \ ((M_cad * u) ./ Δt .+ fs_cad .+ q)
        advance_phases!(Xa, Xm, Tmax, uprev, u, Δt, cad_nd)
        t = tnew
    end
    (; Xa, Xm, Tmax, HV = phase_hardness.(Xa, Xm))
end

# ╔═╡ f12c000e-000e-4a0e-8b0e-00000000000e
md"""
### The hardened part
"""

# ╔═╡ f12c000f-000f-4a0f-8b0f-00000000000f
let
    vf(mask) = round(100 * count(mask) / cad_nd, digits = 2)
    Markdown.parse("""
    | quantity | value |
    |:--|:--|
    | mesh | $(getncells(cad_grid)) tets, $(cad_nd) dofs |
    | exterior / laser-facing facets | $(length(cad_ΓN)) / $(length(cad_Γlaser)) |
    | peak temperature reached | **$(round(maximum(cad_result.Tmax), digits=1)) K** |
    | volume austenitised (Xₐ > 0.5) | $(vf(cad_result.Xa .> 0.5)) % |
    | volume martensitic (Xₘ > 0.5) | $(vf(cad_result.Xm .> 0.5)) % |
    | hardness range | $(round(minimum(cad_result.HV), digits=1)) – $(round(maximum(cad_result.HV), digits=1)) HV |
    | beam dipped below the flat top by | $(round(cad_bb[6] - minimum(lcz), digits=1)) mm (into the groove) |
    """)
end

# ╔═╡ f12c0010-0010-4a10-8b10-000000000010
let
    # interpolate the hardness onto each laser-facing facet centroid → surface map
    fc = FacetCache(cad_dh)
    cx = Float64[]; cy = Float64[]; cz = Float64[]; ch = Float64[]
    for fi in cad_Γlaser
        reinit!(fc, fi); reinit!(fv, fc)
        coords = getcoordinates(fc); nq = getnquadpoints(fv)
        ue = cad_result.HV[celldofs(fc)]
        c  = sum(spatial_coordinate(fv, qp, coords) for qp in 1:nq) / nq
        hv = sum(function_value(fv, qp, ue) for qp in 1:nq) / nq
        push!(cx, c[1]); push!(cy, c[2]); push!(cz, c[3]); push!(ch, hv)
    end
    fig = Mke.Figure(size = (1000, 440))
    ax = Mke.Axis3(fig[1,1], aspect = :data, azimuth = 0.7π, elevation = 0.25π,
                   title = "Hardness on the part surface — the projected, groove-following track")
    sc = Mke.scatter!(ax, cx, cy, cz, color = ch, colormap = :plasma,
                      colorrange = (220, 750), markersize = 9)
    Mke.Colorbar(fig[1,2], sc, label = "HV")
    fig
end

# ╔═╡ f12c0011-0011-4a11-8b11-000000000011
md"""
### Optional — a beam *normal to the surface*

The `n̂·ẑ` factor above assumes the beam always travels **straight down**: a tilted
facet catches less of it, by the cosine of its tilt *from vertical*. That suits a fixed
overhead laser. But a head on a robot arm can **stay perpendicular to the surface** as
it tracks a contour — the energy should then fall off by tilt *relative to the surface
under the spot*, not relative to vertical. The production solver exposes this as a
one-line knob, `LASER_NORMAL_TO_SURFACE`. The only change is the cosine's *reference
direction*:

```
cosθ = n̂ · ẑ                  # vertical beam        (everything above)
cosθ = max(0, n̂ · n̂_ref(t))   # surface-normal beam  (the knob)
```

`n̂_ref(t)` is the outward normal of the facet the moving spot centre is currently over,
found by a **point-in-(x,y)-triangle** test across the up-facing facets (the facet
genuinely beneath the beam — not merely the nearest centroid, which near a steep wall
can be the wrong one). The spot centre then sees **normal incidence** (cosθ = 1) and
the falloff is by tilt *relative to that facet*. Where the projected facet is
horizontal, `n̂_ref = ẑ` and nothing changes — so the knob only bites where the surface
tilts. The plot traces `n̂_ref` along this scan: flat over the top and the groove floor,
swinging steeply up the trough's end-walls.
"""

# ╔═╡ f12c0012-0012-4a12-8b12-000000000012
begin
    # Per laser-facing facet: its (x,y) triangle, (x,y) centroid, and outward normal.
    cadn_tri = NTuple{3,NTuple{2,Float64}}[]
    cadn_cxy = NTuple{2,Float64}[]
    cadn_nrm = Vec{3,Float64}[]
    let fcn = FacetCache(cad_dh)
        for fi in cad_Γlaser
            cell = getcells(cad_grid, fi[1])
            vids = Ferrite.facets(cell)[fi[2]]                      # 3 node ids of this facet
            push!(cadn_tri, ntuple(j -> (cad_grid.nodes[vids[j]].x[1],
                                         cad_grid.nodes[vids[j]].x[2]), 3))
            reinit!(fcn, fi); reinit!(fv, fcn)
            nq = getnquadpoints(fv); cds = getcoordinates(fcn)
            c  = sum(spatial_coordinate(fv, qp, cds) for qp in 1:nq) / nq
            nv = sum(getnormal(fv, qp) for qp in 1:nq) / nq         # planar facet ⇒ constant
            push!(cadn_cxy, (c[1], c[2])); push!(cadn_nrm, nv / norm(nv))
        end
    end

    # 2-D point-in-triangle (sign-consistent; handles either winding)
    function cadn_in_tri(px, py, a, b, c)
        d1 = (px-b[1])*(a[2]-b[2]) - (a[1]-b[1])*(py-b[2])
        d2 = (px-c[1])*(b[2]-c[2]) - (b[1]-c[1])*(py-c[2])
        d3 = (px-a[1])*(c[2]-a[2]) - (c[1]-a[1])*(py-a[2])
        !(((d1<0)|(d2<0)|(d3<0)) & ((d1>0)|(d2>0)|(d3>0)))
    end

    # n̂_ref(t): outward normal of the facet the beam centre projects onto. Point-in-
    # triangle picks the facet genuinely beneath the spot (nearest centroid as fallback).
    function reference_normal_cad(t)
        xc = cad_xspot(t); yc = cad_yscan
        for kk in eachindex(cadn_tri)
            a, b, c = cadn_tri[kk]
            cadn_in_tri(xc, yc, a, b, c) && return cadn_nrm[kk]
        end
        best = 1; bestd = Inf
        for kk in eachindex(cadn_cxy)
            d = (cadn_cxy[kk][1]-xc)^2 + (cadn_cxy[kk][2]-yc)^2
            d < bestd && (bestd = d; best = kk)
        end
        return cadn_nrm[best]
    end
    (laser_facets = length(cadn_tri),)
end

# ╔═╡ f12c0013-0013-4a13-8b13-000000000013
let
    ts   = range(0.0, cad_tscan; length = 300)
    xc   = [cad_xspot(t) for t in ts]
    tilt = [rad2deg(acos(clamp(reference_normal_cad(t)[3], -1, 1))) for t in ts]
    fig = Mke.Figure(size = (900, 320))
    ax = Mke.Axis(fig[1,1], xlabel = "x of beam centre [mm]",
                  ylabel = "n̂_ref tilt from vertical  [°]",
                  title = "Reference normal vs. position along the scan (knob ON)")
    Mke.lines!(ax, xc, tilt, color = :crimson, linewidth = 2)
    Mke.hlines!(ax, [0.0], color = (:gray, 0.5), linestyle = :dash)
    fig
end

# ╔═╡ 04038676-4021-486b-8742-f6c75ce632e7
md"""
## Recap

We built, from scratch, the full chain of a laser-hardening predictor:

1. **Mesh + FEM scaffolding** — tetrahedra, shape functions, dof numbering.
2. **Assembly** — the constant mass `M` and stiffness `K = kG + h·k·Ũ_s`.
3. **The moving laser** — a Gaussian surface flux `q_s(t)`, rebuilt each step.
4. **Radiation** — linearised into the `T³` matrix `R`.
5. **Metallurgy** — JMAK austenite + Koistinen–Marburger martensite + a hardness
   rule of mixtures.
6. **Time marching** — a backward-Euler heat step with an operator-split phase update,
   ending in a map of the **hardened track** and its **case depth**.
7. **Arbitrary geometry** — mesh a real CAD part, recover its outer surface from the
   mesh topology, and **project the laser onto the outermost face** so the beam follows
   a non-convex groove, hardening the part wherever it lands — optionally with the beam
   kept **normal to the local surface** instead of vertical.

The production solver adds an adaptive stiff integrator and multi-core parallelism for
fine meshes, but the construction — and the physics — is exactly what you just ran.
See `laser_hardening_construction.md` for the full written derivation.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Ferrite = "c061ca5d-56c9-439f-9c0e-210fe06d3992"
FerriteGmsh = "4f95f4f8-b27c-4ae5-9a39-ea55e634e36b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
Ferrite = "~1.4.1"
FerriteGmsh = "~1.3.0"
PlutoUI = "~0.7.83"
WGLMakie = "~0.13.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "25df39a3b5be514c1be70ed3d3a44f980c21aa16"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7715e5b2b186c4d9b664d299d2c9e48b9a778c88"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.6.1"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "JSON", "LinearAlgebra", "Markdown", "MbedTLS", "MsgPack", "Observables", "OrderedCollections", "Random", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "bb43f72801f703ad3c66833bd02b8f54c7328238"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "4.2.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "7bc84b769c1d384315e7b5c4ac03a6c303e6cf35"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.8"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3c8a0a9a6d4a10bdfb6b751bd2b6051ed3e25fd4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.127"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "cac41ca6b2d399adfc95e51240566f8a60a80806"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.0+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FLTK_jll]]
deps = ["Artifacts", "Fontconfig_jll", "FreeType2_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "Xorg_libXft_jll", "Xorg_libXinerama_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "72a4842f93e734f378cf381dae2ca4542f019d23"
uuid = "4fce6fc7-ba6a-5f4c-898f-77e99806d6f8"
version = "1.3.8+0"

[[deps.Ferrite]]
deps = ["EnumX", "ForwardDiff", "LinearAlgebra", "NearestNeighbors", "OrderedCollections", "Preferences", "Reexport", "SparseArrays", "Tensors", "WriteVTK"]
git-tree-sha1 = "f61701b7c7b7a5feee1c8a82b390c63dfa4c9da6"
uuid = "c061ca5d-56c9-439f-9c0e-210fe06d3992"
version = "1.4.1"

    [deps.Ferrite.extensions]
    FerriteBlockArrays = "BlockArrays"
    FerriteMetis = "Metis"
    FerriteSparseMatrixCSR = "SparseMatricesCSR"

    [deps.Ferrite.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    Metis = "2679e427-3c69-5b7f-982b-ece356f1e94b"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"

[[deps.FerriteGmsh]]
deps = ["Ferrite", "Gmsh"]
git-tree-sha1 = "e1a87020d81c2095380499582eee11c9d413a36d"
uuid = "4f95f4f8-b27c-4ae5-9a39-ea55e634e36b"
version = "1.3.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "2c5d0b0e12088cde2cf84afb2784415b1ea3dfee"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.1"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLU_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg"]
git-tree-sha1 = "65af046f4221e27fb79b28b6ca89dd1d12bc5ec7"
uuid = "bd17208b-e95e-5925-bf81-e2f59b3e5c61"
version = "9.0.1+0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.3.0+2"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "364685f5ffde25deb1bbcfd5bb278a5c6b7a9b37"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.11"

    [deps.GeometryBasics.extensions]
    ExtentsExt = "Extents"
    GeometryBasicsGeoInterfaceExt = "GeoInterface"
    IntervalSetsExt = "IntervalSets"

    [deps.GeometryBasics.weakdeps]
    Extents = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Gmsh]]
deps = ["gmsh_jll"]
git-tree-sha1 = "6d815101e62722f4e323514c9fc704007d4da2e3"
uuid = "705231aa-382f-11e9-3f0c-b7cb4346fdeb"
version = "0.3.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "aws_c_s3_jll", "dlfcn_win32_jll", "libaec_jll", "mpif_jll"]
git-tree-sha1 = "45337643a2d97262d5fe72ce1f13e8a662d13d62"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "2.1.2+0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "c35847ca5b4997fc8418836354a56c459bcf48d8"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.14.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"
weakdeps = ["ForwardDiff", "Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "921d7e91687e15a2c7c269c226960491fc041832"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.9"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "aa971a09f0f1fe92fe772713a564aa48abe510df"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.3"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LinearElasticity_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "71e8ee0f9fe0e86a8f8c7f28361e5118eab2f93f"
uuid = "18c40d15-f7cd-5a6d-bc92-87468d86c5db"
version = "5.0.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2eefa8baa858871ae7770c98c3c2a7e46daba5b4"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.3+0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MMG_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "LinearElasticity_jll", "Pkg", "SCOTCH_jll"]
git-tree-sha1 = "70a59df96945782bb0d43b56d0fbfdf1ce2e4729"
uuid = "86086c02-e288-5929-a127-40944b0018b7"
version = "5.6.0+0"

[[deps.MPIABI_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9be143b6045719e8fb019d2b3bc2aebad1184fef"
uuid = "b5ada748-db0f-5fc0-8972-9331c762740c"
version = "0.1.5+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "07dbec8aab01696edc0151a401a6cdfe95b9b885"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "5.0.1+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8e98d5d80b87403c311fd51e8455d4546ba7a5f8"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.12"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "675df097f8eeb28998b2cfe3b25655af73d5f7df"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.6+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "efe001e1ee81b8eee0fe7da5a4328fcbbfd6b3aa"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.12"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NearestNeighbors]]
deps = ["AbstractTrees", "Distances", "StaticArrays"]
git-tree-sha1 = "e2c3bba08dd6dedfe17a17889131b885b8c082f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.27"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OCCT_jll]]
deps = ["Artifacts", "FreeType2_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "Xorg_libXft_jll", "Xorg_libXinerama_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b4d728dcd7f9c42862180d2ee881eeb4b471a02a"
uuid = "baad4e97-8daa-5946-aac2-2edac59d34e1"
version = "7.9.3+0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3287ec88df50429a934ebc6cf14606215e27b987"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.33+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4a33fd64a77949468187339d8b10c44a422082f1"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.12+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "6d6c0ca4824268c1a7dca1f4721c535ac63d9074"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.11+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "32b657a0d57c310a1a172bfc8c8cf68c5e674323"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.5"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SCOTCH_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7110b749766853054ce8a2afaa73325d72d32129"
uuid = "a8d0f55d-b80e-548d-aff6-1a04c175f0f9"
version = "6.1.3+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "770240df9a3b8888065046948f7a09b4e0f997d5"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Tensors]]
deps = ["ForwardDiff", "LinearAlgebra", "PrecompileTools", "SIMD", "StaticArrays", "Statistics"]
git-tree-sha1 = "77823f2ca7c8f1405e7037e1f23432e756fbd3f4"
uuid = "48a634ad-e948-5137-8d70-aa71f2a747f4"
version = "1.17.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "57e1b2c9de4bd6f40ecb9de4ac1797b81970d008"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.28.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.VTKBase]]
git-tree-sha1 = "c2d0db3ef09f1942d08ea455a9e252594be5f3b6"
uuid = "4004b06d-e244-455f-a6ce-a5f9919cc534"
version = "1.0.1"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "a21237d60281e607a47bf5835178ee5c13941dae"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.13.12"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.WriteVTK]]
deps = ["Base64", "CodecZlib", "FillArrays", "LightXML", "TranscodingStreams", "VTKBase"]
git-tree-sha1 = "a329e0b6310244173690d6a4dfc6d1141f9b9370"
uuid = "64499a7a-5c06-52f2-abe2-ccb03c286192"
version = "1.21.2"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXft_jll]]
deps = ["Artifacts", "Fontconfig_jll", "JLLWrappers", "Libdl", "Xorg_libXrender_jll"]
git-tree-sha1 = "d893c27836da7986c3248997a2a9535e5e4d8a95"
uuid = "2c808117-e144-5220-80d1-69d4eaa9352c"
version = "2.3.9+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.aws_c_auth_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_http_jll", "aws_c_sdkutils_jll"]
git-tree-sha1 = "8cab83c96af80a1be968251ce1a0548a7545484d"
uuid = "2b3700d1-4306-52e2-a478-c162f0c514be"
version = "0.9.6+0"

[[deps.aws_c_cal_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "22c0f42f4a1f0dc5dcfa8fd267c4ac407c455e7a"
uuid = "70f11efc-bab2-57f1-b0f3-22aad4e67c4b"
version = "0.9.13+0"

[[deps.aws_c_common_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a759cb9bf456ad792cc7898a81ae333cce9ef02a"
uuid = "73048d1d-b8c4-5092-a58d-866c5e8d1e50"
version = "0.12.6+0"

[[deps.aws_c_compression_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "7910c72f45f44afd297c39fe43b99c56d5ed22ec"
uuid = "73a04cd5-f3d7-5bac-9290-e8adb709f224"
version = "0.3.2+0"

[[deps.aws_c_http_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_compression_jll", "aws_c_io_jll"]
git-tree-sha1 = "e358d5a001ef7afbd4f8c5225322512819cda2f2"
uuid = "3254fc65-9028-534d-aa9d-d76d128babc6"
version = "0.10.13+0"

[[deps.aws_c_io_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_common_jll", "s2n_tls_jll"]
git-tree-sha1 = "7e481d474b2087ee8bbf55b81bf9119f21e396d9"
uuid = "13c41daa-f319-5298-b5eb-5754e0170d52"
version = "0.26.3+0"

[[deps.aws_c_s3_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_auth_jll", "aws_c_common_jll", "aws_c_http_jll", "aws_checksums_jll", "s2n_tls_jll"]
git-tree-sha1 = "3e9917ab25114feba657e71be41cad068b9f6595"
uuid = "bd1f34fb-993f-5903-a121-aaf302eed6d4"
version = "0.11.5+0"

[[deps.aws_c_sdkutils_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "c43dfba2c1ab9ea9f02f2c80e86fa16f6460244e"
uuid = "1282aa60-004d-510b-9f52-12498d409daa"
version = "0.2.4+1"

[[deps.aws_checksums_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "2570c8e23f4771a087b12a47edcaaa670ac05a01"
uuid = "b2a88e68-78e7-5e94-8c20-c02986ec140e"
version = "0.2.10+0"

[[deps.dlfcn_win32_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e141d67ffe550eadfb5af1bdbdaf138031e4805f"
uuid = "c4b69c83-5512-53e3-94e6-de98773c479f"
version = "1.4.2+0"

[[deps.gmsh_jll]]
deps = ["Artifacts", "Cairo_jll", "CompilerSupportLibraries_jll", "FLTK_jll", "FreeType2_jll", "GLU_jll", "GMP_jll", "HDF5_jll", "JLLWrappers", "JpegTurbo_jll", "LLVMOpenMP_jll", "Libdl", "Libglvnd_jll", "METIS_jll", "MMG_jll", "OCCT_jll", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "Xorg_libXft_jll", "Xorg_libXinerama_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b81ac8f1fe6a1a473ac9b980de24e9143fa6211"
uuid = "630162c2-fc9b-58b3-9910-8442a8a132e6"
version = "4.15.2+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60f4792734488db6f42e2c7699f1d4594780bd03"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.7+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.mpif_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML"]
git-tree-sha1 = "a8083ee0737c243c8f40a4ba86a0956997facb73"
uuid = "9aeb927a-4695-514f-a259-621a69f20ec0"
version = "0.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.s2n_tls_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "64ae051c6f03044eb7d98027d1b552b4e21e650c"
uuid = "cddc5d3d-934d-5d3a-9747-62fc12ea3f48"
version = "1.7.3+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╟─4a47cd69-8735-4c8d-9079-11a9c18ebb76
# ╠═6683b50e-e01f-4b44-9490-f9c48083328f
# ╟─393c4a7b-57f8-4da9-83ad-bd7361f3504b
# ╟─24111f3d-b0f3-4b38-a257-bd26b2846831
# ╟─c9cf292c-291e-46f0-8661-170d885583ec
# ╠═1145cace-7d90-470a-8f94-1710f67de10a
# ╠═d14d70dc-819e-489f-a5b3-72f8c43318df
# ╟─a4b51bd4-2ccf-4858-b1d4-ddbb5b9ba73e
# ╠═99299eed-c933-4d0e-8f80-a7fde0c4cbe6
# ╟─3085b0eb-1b83-4943-84e9-c2c583c29b63
# ╠═599d208d-5d1d-4277-af4c-f1f62fbe0809
# ╠═fbfc9465-5602-4e96-9a97-70b318f40960
# ╠═3aa2ace5-0ed0-447f-b342-7372026eeffa
# ╠═a0581bd4-a18d-428f-959f-cbc048b38d94
# ╟─97490518-3fc3-4812-bd5c-a150e690063b
# ╠═3a3e9025-77f7-40bb-a571-36fb7b0d9c12
# ╟─7f067a7e-e6a1-4dca-8527-a11209ddec85
# ╠═343aaee6-ce7b-41b4-be32-e35d775c5bd0
# ╟─fb808342-378a-49b6-aa72-bc73010f4c1c
# ╠═b54ac871-c233-4976-99db-34f3033dc364
# ╠═472c663c-e762-4d3d-9bc0-2541b641ec55
# ╟─97188e99-a10b-4955-ad79-f1be3145a71a
# ╠═8dc6a16d-3f94-444c-9670-0c10cd91b475
# ╟─82d3a719-29ed-4611-b9f8-6d378474453a
# ╠═e54429c6-f52b-4e69-bef1-ebe8a180df99
# ╠═dd5ffd5f-f3ab-4c18-a17b-c8542c14d6ec
# ╟─3e705579-dc95-4dcb-8b4d-ee6c53bd5467
# ╠═bb37cb91-3f9c-47f7-866a-cd09d4e677d2
# ╠═04b12484-2ab4-4a4a-adce-e12bfe6dc4f3
# ╟─d5bdf07f-0571-43b5-a0d6-d72ced82f9ba
# ╠═ae1d4676-8228-4cb0-b546-c7711646fcaf
# ╠═7eb91bfa-3737-4b33-abbb-72646a232ccb
# ╠═58e8c1cb-3b83-4233-b4f1-dc74ac95e5ef
# ╟─e4abc8a4-0d67-4c13-9890-25ce9a3f8411
# ╟─0f5681d2-dc94-431c-acf9-022a6e753dc5
# ╟─d3d55413-dee9-470e-aff5-28889fcbf850
# ╠═d459110a-880c-4e4b-b691-be1de65bee3f
# ╠═427c01f3-b39e-4ec7-b986-4140944a7f68
# ╟─18d6d29e-8141-4595-a819-155691b9eaae
# ╠═8ae211dd-b10e-40a3-8964-725db9f8e04d
# ╟─b8f0a112-97a1-4cb5-9c31-8f03255ef815
# ╠═62f210b4-5ab3-4a9d-bd92-42dd89c45051
# ╟─5f087f57-147c-4743-a119-866fef96baef
# ╠═4da1ec68-0f85-4b01-9bb6-ac521daf9884
# ╟─a204af3d-1bf6-459c-9d5b-ad7bea1cba15
# ╠═2c0c84ad-245f-40dd-be88-0299f1f2cc51
# ╟─2fef4b4e-6845-45ab-a56a-5c2d9ef3cf2a
# ╠═a7e0ae6e-7085-482a-867c-e5f68f413c5e
# ╟─f12c0001-0001-4a01-8b01-000000000001
# ╠═f12c0002-0002-4a02-8b02-000000000002
# ╠═f12c0003-0003-4a03-8b03-000000000003
# ╠═f12c0004-0004-4a04-8b04-000000000004
# ╟─f12c0005-0005-4a05-8b05-000000000005
# ╠═f12c0006-0006-4a06-8b06-000000000006
# ╠═f12c0007-0007-4a07-8b07-000000000007
# ╟─f12c0008-0008-4a08-8b08-000000000008
# ╠═f12c0009-0009-4a09-8b09-000000000009
# ╠═f12c000a-000a-4a0a-8b0a-00000000000a
# ╠═f12c000b-000b-4a0b-8b0b-00000000000b
# ╟─f12c000c-000c-4a0c-8b0c-00000000000c
# ╠═f12c000d-000d-4a0d-8b0d-00000000000d
# ╟─f12c000e-000e-4a0e-8b0e-00000000000e
# ╟─f12c000f-000f-4a0f-8b0f-00000000000f
# ╠═f12c0010-0010-4a10-8b10-000000000010
# ╟─f12c0011-0011-4a11-8b11-000000000011
# ╠═f12c0012-0012-4a12-8b12-000000000012
# ╠═f12c0013-0013-4a13-8b13-000000000013
# ╟─04038676-4021-486b-8742-f6c75ce632e7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
