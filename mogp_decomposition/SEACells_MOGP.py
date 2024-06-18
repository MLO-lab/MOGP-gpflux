import SEACells


def SEACells_compute(struct_MOGP, n_SEACells_cells, build_kernel_on, n_waypoint_eigs):
    
    model = SEACells.core.SEACells(struct_MOGP, 
                  build_kernel_on=build_kernel_on, 
                  n_SEACells=n_SEACells_cells, 
                  n_waypoint_eigs=n_waypoint_eigs,
                  convergence_epsilon = 1e-5)
    
    model.construct_kernel_matrix()
    
    M = model.kernel_matrix
    # Initialize archetypes
    model.initialize_archetypes()
    
    # Plot the initilization to ensure they are spread across phenotypic space
    SEACells.plot.plot_initialization(struct_MOGP, model)
    model.fit(min_iter=10, max_iter=100)
    # You can force the model to run additional iterations step-wise using the .step() function
    print(f'Ran for {len(model.RSS_iters)} iterations')
    for _ in range(5):
        model.step()
    print(f'Ran for {len(model.RSS_iters)} iterations')
    # Check for convergence 
    model.plot_convergence()
    SEACells.plot.plot_2D(struct_MOGP, key='X_umap', colour_metacells=True)

    return struct_MOGP
