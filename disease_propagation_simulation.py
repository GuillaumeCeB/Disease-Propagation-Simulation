import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.integrate import odeint

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import IntVar, DoubleVar, StringVar, BooleanVar


class SuperEpidemicModel:
    def __init__(self):
        # Default model parameters
        self.params = {
            'beta': 0.3,         # Base transmission rate
            'gamma': 0.1,        # Base recovery rate
            'xi': 0.05,          # Rate of waning immunity (R->S)
            'alpha': 0.01,       # Disease-induced death rate
            'rho': 0.01,         # Vaccination rate
            'delta': 0.0,        # Waning vaccine protection rate
            'kappa': 0.01,       # Quarantine rate
            'q_eff': 0.5,        # Quarantine effectiveness
            'B': 0,              # Birth rate
            'mu': 0.001,         # Natural death rate
            'm_0': 0.01,         # Base migration rate
            'm_amp': 0.5,        # Migration amplitude (0-1)
            'amp': 0.1,          # Amplitude of seasonal variation
            'phi': 0.0,          # Phase shift for seasonal variation
            'period': 365.0,     # Periodicity in days
            'beta_y': 0.3,       # Transmission rate for young
            'beta_e': 0.2,       # Transmission rate for elderly
            'gamma_y': 0.1,      # Recovery rate for young
            'gamma_e': 0.05,     # Recovery rate for elderly
            'mut_rate': 0.01,    # Rate of mutation per infected individual
            'mut_effect': 0.5,   # Effect of mutation on transmission rate
        }
        
        # Model configuration
        self.model_config = {
            'use_SIRS': True,          # Use SIRS model (waning immunity)
            'use_D': False,            # Include Dead compartment
            'use_V': False,            # Include Vaccinated compartment
            'use_Q': False,            # Include Quarantined compartment
            'seasonal_forcing': False, # Include seasonal variation
            'vital_dynamics': False,   # Include birth and natural death
            'migration': False,        # Include migration
            'age_structure': False,    # Include age structure
            'use_mutation': False,     # Include virus mutation
            'normalize': True,         # Normalize population to sum to 1
        }
        
        # Initial conditions
        self.initial_conditions = {
            'S': 0.99,    # Susceptible
            'I': 0.01,    # Infected
            'R': 0.0,     # Recovered
            'D': 0.0,     # Dead
            'V': 0.01,    # Vaccinated
            'Q': 0.01,    # Quarantined
            'S_y': 0.6,   # Susceptible young
            'I_y': 0.01,  # Infected young
            'R_y': 0.0,   # Recovered young
            'S_e': 0.4,   # Susceptible elderly
            'I_e': 0.0,   # Infected elderly
            'R_e': 0.0,   # Recovered elderly
            'M': 0.0,     # Mutation level (0-1)
        }
        
        # Simulation settings
        # self.simulation_time = 365 * 3  # 3 years
        self.simulation_time = 200
        self.time_points = 1000
        
        # Results storage
        self.last_results = None

    def seasonal_beta(self, t, beta_0, amp, phi, period):
        """Calculate seasonally varying transmission rate with custom periodicity"""
        return beta_0 * (1 + amp * np.cos(2 * np.pi * t / period + phi))

    def migration_rate(self, t, m_0, period):
        """Calculate migration rate with seasonal variation and custom periodicity"""
        # Use m_amp from params to control oscillation amplitude
        m_amp = self.params['m_amp']
        return m_0 * m_amp * np.cos(2 * np.pi * t / period)
    
    def get_initial_conditions(self):
        """Prepare initial conditions array based on current model configuration"""
        config = self.model_config
        
        if config['age_structure']:
            # Age-structured model
            y0 = [
                self.initial_conditions['S_y'],
                self.initial_conditions['I_y'],
                self.initial_conditions['R_y'],
                self.initial_conditions['S_e'],
                self.initial_conditions['I_e'],
                self.initial_conditions['R_e']
            ]
            
            if config['use_mutation']:
                y0.append(self.initial_conditions['M'])
        else:
            # Standard SIR model
            y0 = [
                self.initial_conditions['S'],
                self.initial_conditions['I'],
                self.initial_conditions['R']
            ]
            
            if config['use_D']:
                y0.append(self.initial_conditions['D'])
            if config['use_V']:
                y0.append(self.initial_conditions['V'])
            if config['use_Q']:
                y0.append(self.initial_conditions['Q'])
            if config['use_mutation']:
                y0.append(self.initial_conditions['M'])
        
        return np.array(y0)
    
    def model_equations(self, y, t, params, config):
        """Unified model differential equations"""
        # Extract parameters (done once for efficiency)
        beta = params['beta']
        gamma = params['gamma']
        xi = params['xi']
        alpha = params['alpha']
        rho = params['rho']
        delta = params['delta']
        kappa = params['kappa']
        q_eff = params['q_eff']
        B = params['B']
        mu = params['mu']
        m_0 = params['m_0']
        amp = params['amp']
        phi = params['phi']
        period = params.get('period', 365)
        beta_y = params['beta_y']
        beta_e = params['beta_e']
        gamma_y = params['gamma_y']
        gamma_e = params['gamma_e']
        mut_rate = params['mut_rate']
        mut_effect = params['mut_effect']
        
        # Configuration flags (for cleaner code)
        use_SIRS = config['use_SIRS']
        use_D = config['use_D']
        use_V = config['use_V']
        use_Q = config['use_Q']
        use_seasonal = config['seasonal_forcing']
        use_vital = config['vital_dynamics']
        use_migration = config['migration']
        use_mutation = config['use_mutation']
        age_structure = config['age_structure']
        
        # Initialize derivatives array with zeros (more efficient than appending)
        derivatives = np.zeros_like(y)
        
        # Age-structured model equations
        if age_structure:
            # Extract state variables
            S_y, I_y, R_y, S_e, I_e, R_e = y[:6]
            
            # Extract mutation level if enabled
            M = 0.0

            # Add waning immunity if SIRS model is enabled
            if use_SIRS:
                derivatives[0] += xi * R_y  # R_y -> S_y
                derivatives[2] -= xi * R_y
                derivatives[3] += xi * R_e  # R_e -> S_e
                derivatives[5] -= xi * R_e

            if use_mutation and len(y) > 6:
                M = np.clip(y[6], 0, 1)
                
            # Apply mutation effect to transmission rates if enabled
            b_y = beta_y
            b_e = beta_e
            if use_mutation:
                b_y *= (1 + mut_effect * M)
                b_e *= (1 + mut_effect * M)
                
            # Apply seasonal forcing if enabled
            if use_seasonal:
                seasonal_factor = 1 + amp * np.cos(2 * np.pi * t / period + phi)
                b_y *= seasonal_factor
                b_e *= seasonal_factor
            
            # Calculate basic transmission dynamics
            # Young compartment flows
            S_to_I_y = b_y * S_y * I_y + b_e * S_y * I_e
            I_to_R_y = gamma_y * I_y
            
            # Elderly compartment flows
            S_to_I_e = b_e * S_e * I_e + b_y * S_e * I_y
            I_to_R_e = gamma_e * I_e
            
            # Set basic derivatives
            derivatives[0] = -S_to_I_y  # dS_y/dt
            derivatives[1] = S_to_I_y - I_to_R_y  # dI_y/dt
            derivatives[2] = I_to_R_y  # dR_y/dt
            derivatives[3] = -S_to_I_e  # dS_e/dt
            derivatives[4] = S_to_I_e - I_to_R_e  # dI_e/dt
            derivatives[5] = I_to_R_e  # dR_e/dt
            
            # Apply vital dynamics if enabled
            if use_vital:
                N_total = S_y + I_y + R_y + S_e + I_e + R_e
                
                # Birth and natural death for young
                derivatives[0] += B * N_total * 0.7 - mu * S_y
                derivatives[1] -= mu * I_y
                derivatives[2] -= mu * R_y
                
                # Birth and natural death for elderly
                derivatives[3] += B * N_total * 0.3 - mu * S_e
                derivatives[4] -= mu * I_e
                derivatives[5] -= mu * R_e
            
            # Apply migration if enabled
            if use_migration:
                m = self.migration_rate(t, m_0, period)
                N_total = S_y + I_y + R_y + S_e + I_e + R_e
                
                if abs(m) > 1e-10 and N_total > 0:  # Avoid division by zero
                    if m > 0:  # Immigration
                        # Add immigrants as young and elderly susceptibles with proper ratio
                        derivatives[0] += m * 0.7  # 70% young
                        derivatives[3] += m * 0.3  # 30% elderly
                    else:  # Emigration
                        out_m = abs(m)
                        
                        # Limit emigration to prevent population extinction
                        # Never allow more than 1% of population to leave per day
                        max_emigration = 0.01 * N_total
                        effective_out_m = min(out_m, max_emigration)
                        
                        # Remove proportionally from all compartments
                        derivatives[0] -= effective_out_m * (S_y / N_total)
                        derivatives[1] -= effective_out_m * (I_y / N_total)
                        derivatives[2] -= effective_out_m * (R_y / N_total)
                        derivatives[3] -= effective_out_m * (S_e / N_total)
                        derivatives[4] -= effective_out_m * (I_e / N_total)
                        derivatives[5] -= effective_out_m * (R_e / N_total)
            
            # Calculate mutation dynamics if enabled
            if use_mutation:
                I_total = I_y + I_e
                if len(y) > 6 and y[6] < 1.0:
                    derivatives[6] = mut_rate * I_total * (1 - y[6])
                elif len(y) > 6:
                    derivatives[6] = 0
        
        # Non-age-structured model equations
        else:
            # Extract state variables based on enabled compartments
            S, I, R = y[0], y[1], y[2]
            
            idx = 3
            D = 0
            if use_D:
                D = y[idx]
                idx += 1
                
            V = 0
            if use_V:
                V = y[idx]
                idx += 1
                
            Q = 0
            if use_Q:
                Q = y[idx]
                idx += 1
            
            M = 0.0
            if use_mutation and idx < len(y):
                M = np.clip(y[idx], 0, 1)
            
            # Calculate effective beta (with mutation and seasonal effects)
            beta_effective = beta
            if use_mutation:
                beta_effective *= (1 + mut_effect * M)
                
            if use_seasonal:
                beta_effective = beta * (1 + amp * np.cos(2 * np.pi * t / period + phi))
            
            # Calculate basic infection dynamics
            if use_Q:
                I_effective = I + (1 - q_eff) * Q
                infection_rate = beta_effective * S * I_effective
            else:
                infection_rate = beta_effective * S * I

            # Basic derivatives for SIR model
            derivatives[0] = -infection_rate  # dS/dt
            derivatives[1] = infection_rate - gamma * I  # dI/dt
            derivatives[2] = gamma * I  # dR/dt
            
            # Add waning immunity if SIRS model is enabled
            if use_SIRS:
                derivatives[0] += xi * R  # R -> S
                derivatives[2] -= xi * R
            
            # Modify for quarantine if enabled
            if use_Q:
                idx_Q = 5 if use_D and use_V else (4 if use_D or use_V else 3)
                derivatives[1] -= kappa * I  # Remove quarantined from I
                derivatives[2] = gamma * I + gamma * Q  # Add recovered from Q
                derivatives[idx_Q] = kappa * I - gamma * Q  # dQ/dt
            
            # Add disease-induced mortality if enabled
            if use_D:
                idx_D = 3
                derivatives[1] -= alpha * I  # Remove deaths from I
                derivatives[idx_D] = alpha * I  # dD/dt
            
            # Add vaccination dynamics if enabled
            if use_V:
                idx_V = 4 if use_D else 3
                derivatives[0] -= rho * S  # Vaccinate susceptibles
                derivatives[idx_V] = rho * S - delta * V  # dV/dt
                derivatives[0] += delta * V  # Return to susceptible after waning
            
            # Add vital dynamics (births and deaths) if enabled
            if use_vital:
                N_living = S + I + R + V + Q  # Total living population
                derivatives[0] += B * N_living - mu * S  # Add births, remove natural deaths
                derivatives[1] -= mu * I  # Natural deaths from I
                derivatives[2] -= mu * R  # Natural deaths from R
                
                if use_V:
                    derivatives[idx_V] -= mu * V  # Natural deaths from V
                if use_Q:
                    derivatives[idx_Q] -= mu * Q  # Natural deaths from Q
            
            # Apply migration if enabled
            if use_migration:
                m = self.migration_rate(t, m_0, period)
                N_living = S + I + R
                if use_V:
                    N_living += V
                if use_Q:
                    N_living += Q
                    
                if abs(m) > 1e-10 and N_living > 0:  # Avoid division by zero
                    if m > 0:  # Immigration
                        # New immigrants are susceptible
                        derivatives[0] += m
                    else:  # Emigration
                        out_m = abs(m)
                        
                        # Limit emigration to prevent population extinction
                        # Never allow more than 1% of population to leave per day
                        max_emigration = 0.01 * N_living
                        effective_out_m = min(out_m, max_emigration)
                        
                        # Remove proportionally from all compartments
                        derivatives[0] -= effective_out_m * (S / N_living)
                        derivatives[1] -= effective_out_m * (I / N_living)
                        derivatives[2] -= effective_out_m * (R / N_living)
                        
                        if use_V:
                            derivatives[idx_V] -= effective_out_m * (V / N_living)
                        if use_Q:
                            derivatives[idx_Q] -= effective_out_m * (Q / N_living)
                        if use_D:
                            # Death compartment shouldn't lose people through emigration
                            # No modification to D compartment
                            pass
            
            # Calculate mutation dynamics if enabled
            if use_mutation:
                idx_M = idx
                if idx_M < len(y):
                    if y[idx_M] < 1.0:  # Only increase mutation if not at maximum
                        derivatives[idx_M] = mut_rate * I * (1 - y[idx_M])  # Mutation increases with infected population
                    else:
                        derivatives[idx_M] = 0  # Stop mutation increase at maximum
        

        # Add normalization at each time step if enabled
        if config['normalize']:
            # Normalize the derivatives to maintain total population = 1
            if age_structure:
                population_indices = range(6)  # S_y, I_y, R_y, S_e, I_e, R_e
                total_population = sum(y[i] for i in population_indices)
                total_derivative = sum(derivatives[i] for i in population_indices)
                
                if abs(total_derivative) > 1e-10:  # Only adjust if there is a significant change
                    for i in population_indices:
                        # Correct each derivative to maintain sum = 0
                        derivatives[i] -= (y[i] / total_population) * total_derivative
            else:
                # Standard model normalization
                compartment_indices = [0, 1, 2]  # Start with S, I, R
                if use_V:
                    compartment_indices.append(idx_V)
                if use_Q:
                    compartment_indices.append(idx_Q)
                # Intentionally exclude D from normalization if it exists
                
                total_population = sum(y[i] for i in compartment_indices)
                total_derivative = sum(derivatives[i] for i in compartment_indices)
                
                if abs(total_derivative) > 1e-10:  # Only adjust if there is a significant change
                    for i in compartment_indices:
                        # Correct each derivative to maintain sum = 0
                        derivatives[i] -= (y[i] / total_population) * total_derivative
    
        
        return derivatives
    
    def run_simulation(self):
        """Run the epidemic model simulation"""
        try:
            # Create time points for simulation
            t = np.linspace(0, self.simulation_time, self.time_points)
            
            # Get initial conditions based on current model configuration
            y0 = self.get_initial_conditions()
            
            # Normalize initial conditions if enabled
            if self.model_config['normalize']:
                if self.model_config['age_structure']:
                    # For age structured model, normalize the first 6 components
                    total = sum(y0[:6])
                    if total > 0:
                        y0[:6] = y0[:6] / total
                else:
                    # For standard model, find which compartments to normalize
                    normalize_indices = [0, 1, 2]  # Always include S, I, R
                    idx = 3
                    
                    if self.model_config['use_D']:
                        # Skip D for normalization
                        idx += 1
                        
                    if self.model_config['use_V']:
                        normalize_indices.append(idx)
                        idx += 1
                        
                    if self.model_config['use_Q']:
                        normalize_indices.append(idx)
                        idx += 1
                    
                    # Normalize selected compartments
                    norm_values = [y0[i] for i in normalize_indices]
                    total = sum(norm_values)
                    if total > 0:
                        for i, idx in enumerate(normalize_indices):
                            y0[idx] = norm_values[i] / total
            
            # Solve ODE system
            solution = odeint(self.model_equations, y0, t, args=(self.params, self.model_config))
            
            # Post-process solution to ensure normalization at every time point
            if self.model_config['normalize']:
                for i in range(len(t)):
                    if self.model_config['age_structure']:
                        # Normalize age structured compartments
                        total = sum(solution[i, :6])
                        if total > 0:
                            solution[i, :6] = solution[i, :6] / total
                    else:
                        # For standard model
                        normalize_indices = [0, 1, 2]  # Always S, I, R
                        idx = 3
                        
                        if self.model_config['use_D']:
                            # Skip D for normalization
                            idx += 1
                            
                        if self.model_config['use_V']:
                            normalize_indices.append(idx)
                            idx += 1
                            
                        if self.model_config['use_Q']:
                            normalize_indices.append(idx)
                            idx += 1
                        
                        # Get values to normalize
                        norm_values = [solution[i, j] for j in normalize_indices]
                        total = sum(norm_values)
                        
                        if total > 0:
                            for j, idx in enumerate(normalize_indices):
                                solution[i, idx] = norm_values[j] / total
            
            # Store results for later use
            self.last_results = (t, solution)
            
            return t, solution
        
        except Exception as e:
            print(f"Simulation error: {str(e)}")
            raise

class SuperModelGUI:
    def __init__(self, root):
        """Initialize the GUI for the SuperEpidemicModel"""
        # Setup main window
        self.root = root
        self.root.title("Super Epidemic Model Simulator")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Initialize model
        self.model = SuperEpidemicModel()
        
        # UI variable dictionaries (initialized once)
        self.config_vars = {}    # Model configuration checkboxes
        self.param_vars = {}     # Parameter sliders/entries
        self.initial_vars = {}   # Initial condition fields
        self.plot_vars = {}      # Plot configuration options
        
        # Setup UI structure
        self._create_layout()
        self._setup_ui_components()
        
        # Run initial simulation
        self.run_simulation()
    
    def _create_layout(self):
        """Create the main layout structure"""
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Split into top (content) and bottom (buttons) sections
        self.content_frame = ttk.Frame(self.main_frame)
        self.button_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        self.button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Split content into left (controls) and right (plot) sections
        self.controls_frame = ttk.Frame(self.content_frame, width=400)
        self.plot_frame = ttk.Frame(self.content_frame)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Create notebook for tabbed interface in controls section
        self.notebook = ttk.Notebook(self.controls_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tab frames
        self.config_frame = ttk.Frame(self.notebook, padding=5)
        self.params_frame = ttk.Frame(self.notebook, padding=5)
        self.initial_frame = ttk.Frame(self.notebook, padding=5)
        self.sim_frame = ttk.Frame(self.notebook, padding=5)
        
        # Add tabs to notebook
        self.notebook.add(self.config_frame, text="Model Config")
        self.notebook.add(self.params_frame, text="Parameters")
        self.notebook.add(self.initial_frame, text="Initial Conditions")
        self.notebook.add(self.sim_frame, text="Simulation")
    
    def _create_parameter_controls(self):
        """Create controls for model parameters"""        
        # Seasonal parameters
        if True:  # Create these controls regardless of current settings
            seasonal_frame = ttk.LabelFrame(self.param_frame, text="Seasonal Parameters")
            seasonal_frame.pack(fill="x", expand=True, padx=5, pady=5)
            
            # Amplitude control
            self._create_slider(seasonal_frame, 'amp', 'Amplitude', 0.0, 1.0, 0.01, row=0)
            
            # Phase shift control
            self._create_slider(seasonal_frame, 'phi', 'Phase Shift', 0.0, 2*np.pi, 0.1, row=1)
            
            # Add period control
            self._create_slider(seasonal_frame, 'period', 'Period (days)', 30.0, 730.0, 5.0, row=2)        

    def _setup_ui_components(self):
        """Setup all UI components"""
        self._setup_plot()
        self._setup_config_tab()
        self._setup_params_tab()
        self._setup_initial_tab()
        self._setup_sim_tab()
        self._setup_buttons()

    def _setup_sim_tab(self):
        """Setup the simulation settings tab"""
        # Create frame for simulation settings
        settings_frame = ttk.LabelFrame(self.sim_frame, text="Simulation Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Simulation time setting
        ttk.Label(settings_frame, text="Simulation Duration (days):").grid(
            row=0, column=0, sticky="w", pady=5)
        self.sim_time_var = tk.IntVar(value=self.model.simulation_time)
        sim_time_entry = ttk.Entry(settings_frame, width=10, textvariable=self.sim_time_var)
        sim_time_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Time points setting
        ttk.Label(settings_frame, text="Time Points:").grid(
            row=1, column=0, sticky="w", pady=5)
        self.time_points_var = tk.IntVar(value=self.model.time_points)
        time_points_entry = ttk.Entry(settings_frame, width=10, textvariable=self.time_points_var)
        time_points_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Add plot options
        plot_frame = ttk.LabelFrame(self.sim_frame, text="Plot Options", padding=10)
        plot_frame.pack(fill=tk.X, pady=10)
        
        # Log scale option
        self.plot_vars['log_scale'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(plot_frame, text="Use logarithmic scale", 
                    variable=self.plot_vars['log_scale'], 
                    command=self._update_plot).pack(anchor="w")
    
    def _setup_plot(self):
        """Setup the matplotlib plot area"""
        # Create figure with tight layout
        self.fig = Figure(figsize=(7, 5), dpi=100, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas with toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar (with minimal tools for better performance)
        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)
    
    def _setup_config_tab(self):
        """Setup the model configuration tab"""
        # Define configuration options with descriptions
        config_options = [
            ('use_SIRS', 'Enable waning immunity (SIRS)', ' '),
            ('use_D', 'Include Dead compartment (SIRD)', '(disable Normalization)'),
            ('use_V', 'Include Vaccinated compartment (SIRV)', ' '),
            ('use_Q', 'Include Quarantined compartment (SIRQ)', ' '),
            ('seasonal_forcing', 'Include seasonal variation', ' '),
            ('vital_dynamics', 'Include birth and natural death', '(disable Normalization)'),
            ('migration', 'Include migration', '(disable Normalization)'),
            ('age_structure', 'Include age structure', ' '),
            ('use_mutation', 'Include virus mutation', ' '),
            ('normalize', 'Normalize population', ' '),
        ]
        
        # Create header
        ttk.Label(self.config_frame, text="Model Configuration", 
                 font=("Arial", 12, "bold")).grid(row=0, column=0, 
                 sticky="w", pady=(0, 10), columnspan=2)
        
        # Create variables and checkbuttons
        for i, (key, label, tooltip) in enumerate(config_options):
            row = i + 1
            
            # Create variable
            self.config_vars[key] = tk.BooleanVar(value=self.model.model_config[key])
            
            # Create special command for mutation (needs to redraw plot)
            if key == 'use_mutation':
                cmd = lambda: self._handle_special_toggle('use_mutation')
            elif key == 'age_structure':
                cmd = lambda: self._handle_special_toggle('age_structure')
            else:
                cmd = None
                
            # Create checkbutton
            cb = ttk.Checkbutton(self.config_frame, text=label,
                               variable=self.config_vars[key],
                               command=cmd)
            cb.grid(row=row, column=0, sticky="w", pady=2)
            
            # Add tooltip as text hint
            ttk.Label(self.config_frame, text=tooltip, 
                    foreground="gray", font=("Arial", 9)).grid(
                    row=row, column=1, sticky="w", padx=(10, 0))
    
    def _setup_params_tab(self):
        """Setup the parameter control tab"""
        # Create a canvas with scrollbar for many parameters
        canvas = tk.Canvas(self.params_frame)
        scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Group parameters by category for better organization
        param_groups = [
            ("Basic SIR Parameters", [
                ('beta', 'Transmission rate (β)', 0.001, 1.0),
                ('gamma', 'Recovery rate (γ)', 0.001, 1.0),
            ]),
            ("SIRS Parameters", [
                ('xi', 'Waning immunity rate (ξ)', 0.0, 0.1),
            ]),
            ("Disease Mortality", [
                ('alpha', 'Disease mortality (α)', 0.0, 0.5),
            ]),
            ("Vaccination Parameters", [
                ('rho', 'Vaccination rate (ρ)', 0.0, 0.5),
                ('delta', 'Waning vaccine immunity rate (δ)', 0.0, 0.5),
            ]),
            ("Quarantine Parameters", [
                ('kappa', 'Quarantine rate (κ)', 0.0, 0.5),
                ('q_eff', 'Quarantine effectiveness (q_eff)', 0.0, 1.0),
            ]),
            ("Vital Dynamics", [
                ('B', 'Birth rate (B)', 0.0, 0.001),
                ('mu', 'Natural death rate (μ)', 0.0, 0.001),
            ]),
            ("Seasonal Forcing", [
                ('amp', 'Amplitude of seasonality', 0.0, 0.5),
                ('phi', 'Phase shift', 0.0, 5),
            ]),
            ("Migration", [
                ('m_0', 'Base migration rate (m_0)', 0, 0.1),
                ('m_amp', 'Migration amplitude (m_amp)', 0, 1),
            ]),
            ("Age Structure", [
                ('beta_y', 'Young transmission rate (beta_y)', 0.001, 1.0),
                ('beta_e', 'Elderly transmission rate (beta_e)', 0.001, 1.0),
                ('gamma_y', 'Young recovery rate (gamma_y)', 0.001, 1.0),
                ('gamma_e', 'Elderly recovery rate (gamma_e)', 0.001, 1.0),
            ]),
            ("Mutation Parameters", [
                ('mut_rate', 'Mutation rate (mut_rate)', 0.0, 0.1),
                ('mut_effect', 'Mutation effect on transmission (mut_effect)', 0.0, 2.0),
            ])
        ]
        
        # Create all parameters UI
        row = 0
        for group_name, params in param_groups:
            # Add header for each group
            ttk.Label(scrollable_frame, text=group_name, 
                     font=("Arial", 10, "bold")).grid(
                     row=row, column=0, sticky="w", pady=(10, 5), columnspan=3)
            row += 1
            
            # Add each parameter in the group
            for param_key, param_label, min_val, max_val in params:
                ttk.Label(scrollable_frame, text=param_label).grid(
                    row=row, column=0, sticky="w", pady=2)
                
                # Create linked variable
                self.param_vars[param_key] = tk.DoubleVar(value=self.model.params[param_key])
                
                # Create slider and entry linked to same variable
                entry = ttk.Entry(scrollable_frame, width=8, 
                                textvariable=self.param_vars[param_key])
                entry.grid(row=row, column=1, padx=5, pady=2)
                
                slider = ttk.Scale(scrollable_frame, from_=min_val, to=max_val,
                                 orient="horizontal", variable=self.param_vars[param_key],
                                 length=150)
                slider.grid(row=row, column=2, padx=5, pady=2, sticky="ew")
                
                row += 1
    
    def _setup_initial_tab(self):
        """Setup the initial conditions tab"""
        # Create a canvas with scrollbar
        canvas = tk.Canvas(self.initial_frame)
        scrollbar = ttk.Scrollbar(self.initial_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Group initial conditions by category
        initial_groups = [
            ("Basic SIR Model", [
                ('S', 'Initial Susceptible (S)', 0.0, 1.0),
                ('I', 'Initial Infected (I)', 0.0, 1.0),
                ('R', 'Initial Recovered (R)', 0.0, 1.0),
            ]),
            ("Additional Compartments", [
                ('D', 'Initial Dead (D)', 0.0, 1.0),
                ('V', 'Initial Vaccinated (V)', 0.0, 1.0),
                ('Q', 'Initial Quarantined (Q)', 0.0, 1.0),
            ]),
            ("Age Structure Model", [
                ('S_y', 'Initial Susceptible Young', 0.0, 1.0),
                ('I_y', 'Initial Infected Young', 0.0, 1.0),
                ('R_y', 'Initial Recovered Young', 0.0, 1.0),
                ('S_e', 'Initial Susceptible Elderly', 0.0, 1.0),
                ('I_e', 'Initial Infected Elderly', 0.0, 1.0),
                ('R_e', 'Initial Recovered Elderly', 0.0, 1.0),
            ]),
            ("Mutation", [
                ('M', 'Initial Mutation Level', 0.0, 1.0),
            ]),
        ]
        
        # Create all initial condition UI controls
        row = 0
        for group_name, initials in initial_groups:
            # Add group header
            ttk.Label(scrollable_frame, text=group_name,
                     font=("Arial", 10, "bold")).grid(
                     row=row, column=0, sticky="w", pady=(10, 5), columnspan=3)
            row += 1
            
            # Add each initial condition in the group
            for init_key, init_label, min_val, max_val in initials:
                ttk.Label(scrollable_frame, text=init_label).grid(
                    row=row, column=0, sticky="w", pady=2)
                
                # Create linked variable
                self.initial_vars[init_key] = tk.DoubleVar(value=self.model.initial_conditions[init_key])
                
                # Create linked entry and slider
                entry = ttk.Entry(scrollable_frame, width=8,
                                textvariable=self.initial_vars[init_key])
                entry.grid(row=row, column=1, padx=5, pady=2)
                
                slider = ttk.Scale(scrollable_frame, from_=min_val, to=max_val,
                                 orient="horizontal", variable=self.initial_vars[init_key],
                                 length=150)
                slider.grid(row=row, column=2, padx=5, pady=2, sticky="ew")
                
                row += 1
    
    def _setup_buttons(self):
        """Setup control buttons at the bottom"""
        ttk.Button(self.button_frame, text="Run Simulation", 
                 command=self.run_simulation, style="Accent.TButton").pack(
                 side=tk.RIGHT, padx=5)
        
        ttk.Button(self.button_frame, text="Reset Parameters", 
                 command=self._reset_parameters).pack(side=tk.LEFT, padx=5)
    
    def _handle_special_toggle(self, option_key):
        """Handle special toggles that need additional processing"""
        # Update model configuration
        self.model.model_config[option_key] = self.config_vars[option_key].get()
        
        # Run simulation to update the plot
        self.run_simulation()
    
    def _update_model_from_ui(self):
        """Update model parameters from UI control values"""
        # Update configuration
        for key, var in self.config_vars.items():
            self.model.model_config[key] = var.get()
        
        # Update parameters
        for key, var in self.param_vars.items():
            self.model.params[key] = var.get()
        
        # Update initial conditions
        for key, var in self.initial_vars.items():
            self.model.initial_conditions[key] = var.get()
        
        # Update simulation settings
        self.model.simulation_time = self.sim_time_var.get()
        self.model.time_points = self.time_points_var.get()
    
    def run_simulation(self):
        """Run the simulation and update the plot"""
        try:
            # Update model from UI
            self._update_model_from_ui()
            
            # Run simulation
            t, solution = self.model.run_simulation()
            
            # Update plot
            self._update_plot()
            
            return True
        except Exception as e:
            self._show_error(f"Error running simulation: {str(e)}")
            print(f"Simulation error: {e}")
            traceback.print_exc()
            return False
    
    def _update_plot(self):
        """Update the plot with current simulation results"""
        if self.model.last_results is None:
            return
            
        t, solution = self.model.last_results
        
        # Clear existing plot
        self.ax.clear()
        
        # Remove any secondary y-axes
        for axis in self.fig.axes:
            if axis != self.ax:
                axis.remove()
        
        # Create second axis for mutation level if needed
        ax2 = None
        if self.model.model_config['use_mutation']:
            ax2 = self.ax.twinx()
        
        # Set log scale if enabled
        if self.plot_vars['log_scale'].get():
            self.ax.set_yscale('log')
            # Set minimum to small value to avoid log(0)
            self.ax.set_ylim(bottom=1e-6)
        else:
            self.ax.set_yscale('linear')
            self.ax.set_ylim(0, 1.1)
        
        try:
            # Plot results based on model configuration
            if self.model.model_config['age_structure']:
                # Plot age-structured results
                self.ax.plot(t, solution[:, 0], 'c-', label='Susceptible Young (S_y)')
                self.ax.plot(t, solution[:, 1], 'r-', label='Infected Young (I_y)')
                self.ax.plot(t, solution[:, 2], 'g-', label='Recovered Young (R_y)')
                self.ax.plot(t, solution[:, 3], 'c--', label='Susceptible Elderly (S_e)')
                self.ax.plot(t, solution[:, 4], 'r--', label='Infected Elderly (I_e)')
                self.ax.plot(t, solution[:, 5], 'g--', label='Recovered Elderly (R_e)')
                
                # Add total infected line
                total_infected = solution[:, 1] + solution[:, 4]
                self.ax.plot(t, total_infected, 'k-', linewidth=2, label='Total Infected')
                
                # Plot mutation on secondary axis if enabled
                if self.model.model_config['use_mutation'] and ax2 is not None and solution.shape[1] > 6:
                    ax2.plot(t, solution[:, 6], 'b--', linewidth=2, label='Mutation Level')
                    ax2.set_ylabel('Mutation Level')
                    ax2.set_ylim(0, 1.1)
                
            else:
                # Plot standard model results
                idx = 0
                colors = ['blue', 'red', 'green', 'black', 'magenta', 'orange', 'purple']
                labels = ['Susceptible (S)', 'Infected (I)', 'Recovered (R)', 
                          'Dead (D)', 'Vaccinated (V)', 'Quarantined (Q)']
                
                # Basic SIR
                for i in range(3):
                    self.ax.plot(t, solution[:, i], color=colors[i], label=labels[i])
                idx = 3
                
                # Additional compartments
                if self.model.model_config['use_D']:
                    self.ax.plot(t, solution[:, idx], color=colors[3], label=labels[3])
                    idx += 1
                
                if self.model.model_config['use_V']:
                    self.ax.plot(t, solution[:, idx], color=colors[4], label=labels[4])
                    idx += 1
                
                if self.model.model_config['use_Q']:
                    self.ax.plot(t, solution[:, idx], color=colors[5], label=labels[5])
                    idx += 1
                
                # Plot mutation on secondary axis if enabled
                if self.model.model_config['use_mutation'] and ax2 is not None and idx < solution.shape[1]:
                    ax2.plot(t, solution[:, idx], 'b--', linewidth=2, label='Mutation Level')
                    ax2.set_ylabel('Mutation Level')
                    ax2.set_ylim(0, 1.1)
            
            # Set plot labels and grid
            self.ax.set_xlabel('Time (days)')
            self.ax.set_ylabel('Population Fraction')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add title with simulation details
            title = "Epidemic Model Simulation"
            if self.model.model_config['seasonal_forcing']:
                title += " (Seasonal)"
            if self.model.model_config['age_structure']:
                title += " - Age-Structured"
            if self.model.model_config['use_mutation']:
                title += " with Mutation"
            self.ax.set_title(title)
            
            # Add legend
            if ax2 is not None:
                # Combine legends from both axes
                lines1, labels1 = self.ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                self.ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                self.ax.legend(loc='upper right')
            
            # Update canvas
            self.canvas.draw_idle()
            
        except Exception as e:
            self._show_error(f"Error plotting results: {str(e)}")
            print(f"Plot error: {e}")
            traceback.print_exc()
    
    def _save_results(self):
        """Save simulation results to CSV file"""
        if self.model.last_results is None:
            self._show_error("No simulation results to save.")
            return
            
        try:
            # Ask for file path
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
                
            t, solution = self.model.last_results
            
            # Create column headers
            headers = ['Time']
            
            if self.model.model_config['age_structure']:
                headers.extend(['S_y', 'I_y', 'R_y', 'S_e', 'I_e', 'R_e'])
                if self.model.model_config['use_mutation'] and solution.shape[1] > 6:
                    headers.append('M')
            else:
                headers.extend(['S', 'I', 'R'])
                idx = 3
                
                if self.model.model_config['use_D']:
                    headers.append('D')
                    idx += 1
                    
                if self.model.model_config['use_V']:
                    headers.append('V')
                    idx += 1
                    
                if self.model.model_config['use_Q']:
                    headers.append('Q')
                    idx += 1
                    
                if self.model.model_config['use_mutation'] and idx < solution.shape[1]:
                    headers.append('M')
            
            # Combine time and solution data
            data = np.column_stack((t.reshape(-1, 1), solution))
            
            # Save to CSV
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(data)
                
            messagebox.showinfo("Save Successful", f"Results saved to {file_path}")
            
        except Exception as e:
            self._show_error(f"Error saving results: {str(e)}")
    
    def _reset_parameters(self):
        """Reset parameters to default values"""
        try:
            # Create confirmation dialog
            if not messagebox.askyesno("Confirm Reset", "Reset all parameters to default values?"):
                return
                
            # Reset model
            self.model = SuperEpidemicModel()
            
            # Update UI variables
            for key, var in self.config_vars.items():
                var.set(self.model.model_config[key])
                
            for key, var in self.param_vars.items():
                var.set(self.model.params[key])
                
            for key, var in self.initial_vars.items():
                var.set(self.model.initial_conditions[key])
                
            self.sim_time_var.set(self.model.simulation_time)
            self.time_points_var.set(self.model.time_points)
            
            # Run simulation with defaults
            self.run_simulation()
            
            messagebox.showinfo("Reset Complete", "Parameters reset to default values")
            
        except Exception as e:
            self._show_error(f"Error resetting parameters: {str(e)}")
    
    def _show_error(self, message):
        """Show error message in a dialog"""
        messagebox.showerror("Error", message)
    
    # Preset methods
    def _preset_basic_sir(self):
        """Load basic SIR model preset"""
        # Disable all features
        for key in self.config_vars:
            self.config_vars[key].set(False)
        
        # Just keep normalization
        self.config_vars['normalize'].set(True)
        
        # Set SIR parameters
        self.param_vars['beta'].set(0.3)
        self.param_vars['gamma'].set(0.1)
        
        # Set initial conditions
        self.initial_vars['S'].set(0.99)
        self.initial_vars['I'].set(0.01)
        self.initial_vars['R'].set(0.0)
        
        # Set simulation time
        self.sim_time_var.set(200)
        
        # Run simulation
        self.run_simulation()
    
    def _preset_sir_with_mortality(self):
        """Load SIR with mortality preset"""
        self._preset_basic_sir()  # Start with basic SIR
        
        # Enable mortality
        self.config_vars['use_D'].set(True)
        
        # Set mortality rate
        self.param_vars['alpha'].set(0.02)
        
        # Run simulation
        self.run_simulation()
    
    def _preset_seasonal_flu(self):
        """Load seasonal flu preset"""
        # Reset model first
        self._preset_basic_sir()
        
        # Enable seasonal forcing
        self.config_vars['seasonal_forcing'].set(True)
        self.config_vars['vital_dynamics'].set(True)
        
        # Set parameters
        self.param_vars['beta'].set(0.5)
        self.param_vars['gamma'].set(0.1)
        self.param_vars['amp'].set(0.4)
        self.param_vars['phi'].set(0.0)  # Northern hemisphere: peak in winter
        self.param_vars['B'].set(0.00005)
        self.param_vars['mu'].set(0.00005)
        
        # Set simulation time (3 years)
        self.sim_time_var.set(365 * 3)
        
        # Run simulation
        self.run_simulation()
    
    def _preset_covid(self):
        """Load COVID-like preset"""
        # Reset model first
        self._preset_basic_sir()
        
        # Enable features
        self.config_vars['use_D'].set(True)
        self.config_vars['use_Q'].set(True)
        
        # Set parameters
        self.param_vars['beta'].set(0.3)
        self.param_vars['gamma'].set(0.05)
        self.param_vars['alpha'].set(0.01)
        self.param_vars['kappa'].set(0.1)
        self.param_vars['q_eff'].set(0.8)
        
        # Set simulation time
        self.sim_time_var.set(365)
        
        # Run simulation
        self.run_simulation()
    
    def _preset_age_structured(self):
        """Load age-structured preset"""
        # Reset model first
        self._preset_basic_sir()
        
        # Enable age structure
        self.config_vars['age_structure'].set(True)
        
        # Set parameters
        self.param_vars['beta_y'].set(0.4)
        self.param_vars['beta_e'].set(0.2)
        self.param_vars['gamma_y'].set(0.12)
        self.param_vars['gamma_e'].set(0.07)
        
        # Set initial conditions
        self.initial_vars['S_y'].set(0.7)
        self.initial_vars['I_y'].set(0.01)
        self.initial_vars['R_y'].set(0.0)
        self.initial_vars['S_e'].set(0.29)
        self.initial_vars['I_e'].set(0.0)
        self.initial_vars['R_e'].set(0.0)
        
        # Set simulation time
        self.sim_time_var.set(200)
        
        # Run simulation
        self.run_simulation()
    
    def _preset_mutation(self):
        """Load mutation preset"""
        # Reset model first
        self._preset_basic_sir()
        
        # Enable mutation
        self.config_vars['use_mutation'].set(True)
        
        # Set parameters
        self.param_vars['beta'].set(0.2)
        self.param_vars['gamma'].set(0.1)
        self.param_vars['mut_rate'].set(0.02)
        self.param_vars['mut_effect'].set(2.0)
        
        # Set initial conditions
        self.initial_vars['S'].set(0.99)
        self.initial_vars['I'].set(0.01)
        self.initial_vars['R'].set(0.0)
        self.initial_vars['M'].set(0.0)
        
        # Set simulation time
        self.sim_time_var.set(365)
        
        # Run simulation
        self.run_simulation()
    
    def _handle_special_toggle(self, key):
        """Handle special toggle cases that need additional updates"""
        if key == 'use_mutation':
            # Immediately update the model config value
            self.model.model_config['use_mutation'] = self.config_vars['use_mutation'].get()
            
            # If turning on mutation, ensure initial condition exists
            if self.model.model_config['use_mutation']:
                if 'M' not in self.initial_vars:
                    self.initial_vars['M'] = tk.DoubleVar(value=0.0)
            
            # Update simulation
            self.run_simulation()
            
        elif key == 'age_structure':
            # Immediately update the model config value
            self.model.model_config['age_structure'] = self.config_vars['age_structure'].get()
            
            # Update the active tab based on structure type
            if self.model.model_config['age_structure']:
                # Switch to age-structured initial conditions tab if available
                if hasattr(self, 'age_tab'):
                    self.notebook.select(self.age_tab)
            else:
                # Switch to standard initial conditions tab
                if hasattr(self, 'standard_tab'):
                    self.notebook.select(self.standard_tab)
            
            # Update simulation
            self.run_simulation()
    
    def run_simulation(self):
        """Run the simulation and update the plot"""
        try:
            # Update model from UI controls
            self._update_model_from_ui()
            
            # Run simulation
            t, solution = self.model.run_simulation()
            
            # Update plot
            self._plot_results(t, solution)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_error(f"Error running simulation: {str(e)}")
    
    def _update_model_from_ui(self):
        """Update model settings from UI controls"""
        # Update model configuration (boolean values)
        for key, var in self.config_vars.items():
            self.model.model_config[key] = var.get()
        
        # Update model parameters (numeric values) with validation
        for key, var in self.param_vars.items():
            try:
                value = float(var.get())
                # Parameter-specific validation
                if key == 'beta' and value <= 0:
                    messagebox.showwarning("Invalid Value", 
                                        f"Transmission rate (beta) must be greater than 0. Using default value.")
                    var.set(self.model.params[key])
                    continue
                elif key in ['gamma', 'gamma_y', 'gamma_e'] and value <= 0:
                    messagebox.showwarning("Invalid Value", 
                                        f"Recovery rate ({key}) must be greater than 0. Using default value.")
                    var.set(self.model.params[key])
                    continue
                elif key in ['alpha', 'rho', 'delta', 'kappa', 'B', 'mu', 'm_0', 'mut_rate'] and value < 0:
                    messagebox.showwarning("Invalid Value", 
                                        f"Parameter {key} cannot be negative. Using default value.")
                    var.set(self.model.params[key])
                    continue
                elif key in ['q_eff', 'amp'] and (value < 0 or value > 1):
                    messagebox.showwarning("Invalid Value", 
                                        f"Parameter {key} must be between 0 and 1. Using default value.")
                    var.set(self.model.params[key])
                    continue
                
                self.model.params[key] = value
                
                # Simplify model based on parameter values
                self._simplify_model_by_parameter(key, value)
                    
            except ValueError:
                # Handle potential parsing errors
                messagebox.showwarning("Invalid Value", 
                                    f"Invalid value for {key}. Using default.")
                var.set(self.model.params[key])  # Reset to model default
        
        # Update initial conditions (numeric values)
        for key, var in self.initial_vars.items():
            try:
                value = float(var.get())
                if value < 0:
                    messagebox.showwarning("Invalid Value", 
                                        f"Initial condition {key} cannot be negative. Using default value.")
                    var.set(self.model.initial_conditions[key])
                    continue
                self.model.initial_conditions[key] = value
            except ValueError:
                # Handle potential parsing errors
                messagebox.showwarning("Invalid Value", 
                                    f"Invalid value for {key}. Using default.")
                var.set(self.model.initial_conditions[key])  # Reset to model default
        
        # Update simulation settings
        try:
            sim_time = int(self.sim_time_var.get())
            time_points = int(self.time_points_var.get())
            
            if sim_time <= 0:
                messagebox.showwarning("Invalid Value", 
                                    "Simulation time must be positive. Using default.")
                self.sim_time_var.set(self.model.simulation_time)
            else:
                self.model.simulation_time = sim_time
                
            if time_points <= 1:
                messagebox.showwarning("Invalid Value", 
                                    "Number of time points must be greater than 1. Using default.")
                self.time_points_var.set(self.model.time_points)
            else:
                self.model.time_points = time_points
                
        except ValueError:
            # Handle potential parsing errors
            messagebox.showwarning("Invalid Value", 
                                "Invalid simulation settings. Using defaults.")
            self.sim_time_var.set(self.model.simulation_time)
            self.time_points_var.set(self.model.time_points)

    def _simplify_model_by_parameter(self, key, value):
        """Simplify model based on parameter values"""
        # If parameter is 0, disable corresponding model feature
        if key == 'alpha' and value == 0:
            self.model.model_config['use_D'] = False
            self.config_vars['use_D'].set(False)
        elif key == 'rho' and value == 0:
            self.model.model_config['use_V'] = False
            self.config_vars['use_V'].set(False)
        elif key == 'kappa' and value == 0:
            self.model.model_config['use_Q'] = False
            self.config_vars['use_Q'].set(False)
        elif key == 'amp' and value == 0:
            self.model.model_config['seasonal_forcing'] = False
            self.config_vars['seasonal_forcing'].set(False)
        elif (key == 'B' or key == 'mu') and value == 0:
            # Only disable vital dynamics if both birth and death rates are zero
            if key == 'B' and self.model.params['mu'] == 0:
                self.model.model_config['vital_dynamics'] = False
                self.config_vars['vital_dynamics'].set(False)
            elif key == 'mu' and self.model.params['B'] == 0:
                self.model.model_config['vital_dynamics'] = False
                self.config_vars['vital_dynamics'].set(False)
        elif key == 'm_0' and value == 0:
            self.model.model_config['migration'] = False
            self.config_vars['migration'].set(False)
        elif key == 'mut_rate' and value == 0:
            self.model.model_config['use_mutation'] = False
            self.config_vars['use_mutation'].set(False)
    
    def _plot_age_structured(self, t, solution, ax2=None):
        """Plot age-structured model results"""
        # Plot main compartments for young population
        self.ax.plot(t, solution[:, 0], 'b-', linewidth=2, label='S_y (Susceptible Young)')
        self.ax.plot(t, solution[:, 1], 'r-', linewidth=2, label='I_y (Infected Young)')
        self.ax.plot(t, solution[:, 2], 'g-', linewidth=2, label='R_y (Recovered Young)')
        
        # Plot main compartments for elderly population
        self.ax.plot(t, solution[:, 3], 'b--', linewidth=2, label='S_e (Susceptible Elderly)')
        self.ax.plot(t, solution[:, 4], 'r--', linewidth=2, label='I_e (Infected Elderly)')
        self.ax.plot(t, solution[:, 5], 'g--', linewidth=2, label='R_e (Recovered Elderly)')
        
        # Plot total infected for emphasis
        total_infected = solution[:, 1] + solution[:, 4]
        self.ax.plot(t, total_infected, 'k-', linewidth=3, label='Total Infected')
        
        # Plot mutation level if enabled
        if self.model.model_config['use_mutation'] and ax2 is not None and solution.shape[1] > 6:
            ax2.plot(t, solution[:, 6], 'purple', linewidth=2, label='Mutation Level')
            ax2.yaxis.set_ticks([])
            ax2.set_yticklabels([])
            ax2.set_ylim(0, 1.05)
    
    def _plot_standard_model(self, t, solution, ax2=None):
        """Plot standard (non-age-structured) model results"""
        # Track current column index
        idx = 0
        
        # Always plot S, I, R
        self.ax.plot(t, solution[:, idx], 'b-', linewidth=2, label='S (Susceptible)'); idx += 1
        self.ax.plot(t, solution[:, idx], 'r-', linewidth=2, label='I (Infected)'); idx += 1
        self.ax.plot(t, solution[:, idx], 'g-', linewidth=2, label='R (Recovered)'); idx += 1
        
        # Plot additional compartments if enabled
        if self.model.model_config['use_D']:
            self.ax.plot(t, solution[:, idx], 'k-', linewidth=2, label='D (Deaths)'); idx += 1
            
        if self.model.model_config['use_V']:
            self.ax.plot(t, solution[:, idx], 'm-', linewidth=2, label='V (Vaccinated)'); idx += 1
            
        if self.model.model_config['use_Q']:
            self.ax.plot(t, solution[:, idx], 'y-', linewidth=2, label='Q (Quarantined)'); idx += 1
        
        # Plot mutation level if enabled
        if self.model.model_config['use_mutation'] and ax2 is not None and idx < solution.shape[1]:
            ax2.plot(t, solution[:, idx], 'purple', linewidth=2, label='Mutation Level')
            ax2.yaxis.set_ticks([])
            ax2.set_yticklabels([])
            ax2.set_ylim(0, 1.05)
    
    def _plot_results(self, t, solution):
        """Plot simulation results with optimized rendering"""
        # Clear existing plot and axes
        self.ax.clear()
        
        # Remove any secondary y-axes that might exist
        for axis in self.fig.axes:
            if axis != self.ax:
                axis.remove()
        
        # Create secondary y-axis for mutation if needed
        ax2 = None
        if self.model.model_config['use_mutation']:
            ax2 = self.ax.twinx()
        
        # Plot results based on model configuration
        try:
            if self.model.model_config['age_structure']:
                # Age-structured model plot
                self._plot_age_structured(t, solution, ax2)
            else:
                # Standard model plot
                self._plot_standard_model(t, solution, ax2)
            
            # Configure plot styling (common for all plots)
            self._configure_plot_style(ax2)
            
            # Add parameter values to plot
            self._add_parameter_info()
            
            # Update canvas with minimal redraws
            self.canvas.draw_idle()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_error(f"Error plotting results: {str(e)}")

    def _add_parameter_info(self):
        """Add key parameter information to the plot"""
        # Create parameter info text
        param_text = "Parameters:\n"
        
        # Add basic parameters that are always relevant
        param_text += f"β={self.model.params['beta']:.3f}, "
        param_text += f"γ={self.model.params['gamma']:.3f}"

        # Add waning immunity rate if SIRS is enabled
        if self.model.model_config['use_SIRS']:
            param_text += f", ξ={self.model.params['xi']:.3f}"
        
        # Add additional parameters based on model configuration
        if self.model.model_config['use_D']:
            param_text += f", α={self.model.params['alpha']:.3f}"
        
        if self.model.model_config['use_V']:
            param_text += f", ρ={self.model.params['rho']:.3f}"
            param_text += f", δ={self.model.params['delta']:.3f}"
        
        if self.model.model_config['use_Q']:
            param_text += f", κ={self.model.params['kappa']:.3f}"
            param_text += f", q_eff={self.model.params['q_eff']:.2f}"
        
        # Seasonal forcing parameters
        if self.model.model_config['seasonal_forcing']:
            param_text += f"\namp={self.model.params['amp']:.2f}"
            param_text += f", φ={self.model.params['phi']:.2f}"
            param_text += f", period={self.model.params['period']:.1f} days"
        
        # Migration parameter
        if self.model.model_config['migration']:
            param_text += f", m₀={self.model.params['m_0']:.3f}"
            param_text += f", m_amp={self.model.params['m_amp']:.3f}"
        
        # Vital dynamics parameters
        if self.model.model_config['vital_dynamics']:
            param_text += f", B={self.model.params['B']:.5f}"
            param_text += f", μ={self.model.params['mu']:.5f}"
        
        # Mutation parameters
        if self.model.model_config['use_mutation']:
            param_text += f"\nmut_rate={self.model.params['mut_rate']:.3f}"
            param_text += f", mut_effect={self.model.params['mut_effect']:.2f}"
        
        # Age structure parameters
        if self.model.model_config['age_structure']:
            param_text += f"\nβ_y={self.model.params['beta_y']:.3f}"
            param_text += f", β_e={self.model.params['beta_e']:.3f}"
            param_text += f", γ_y={self.model.params['gamma_y']:.3f}"
            param_text += f", γ_e={self.model.params['gamma_e']:.3f}"
        
        # Add text to plot (in upper left, with small font and semi-transparent box)
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        self.ax.text(0.02, 0.98, param_text, transform=self.ax.transAxes,
                    fontsize=8, verticalalignment='top', bbox=bbox_props)

    def _configure_plot_style(self, ax2=None):
        """Configure plot styling and labels"""
        # Set labels and title
        self.ax.set_xlabel('Time (days)', fontsize=10)
        self.ax.set_ylabel('Population Fraction', fontsize=10)
        
        # Create title with basic model info
        title = 'Epidemic Model Simulation: '
        
        # Add model type indicators to title
        model_types = []
        if self.model.model_config['age_structure']:
            model_types.append('Age-Structured')
        if self.model.model_config['use_mutation']:
            model_types.append('With Mutation')
        if self.model.model_config['seasonal_forcing']:
            model_types.append('Seasonal')
        
        # Add basic compartments
        compartments = ['SIR']
        if self.model.model_config['use_SIRS']:
            compartments.append('S')
        if self.model.model_config['use_D']:
            compartments.append('D')
        if self.model.model_config['use_V']:
            compartments.append('V')
        if self.model.model_config['use_Q']:
            compartments.append('Q')
        
        
        # Add model types and compartments to title
        if model_types:
            title += ', '.join(model_types) + ' '
        title += ''.join(compartments) + ' Model'
        
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Set y-axis limits
        if self.model.model_config['normalize']:
            self.ax.set_ylim(0, 1.05)
        
        # Add grid for readability
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Configure legends - place in upper right to avoid parameter text
        if ax2 is not None:
            # Combined legend with both axes
            lines1, labels1 = self.ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        else:
            # Standard legend
            self.ax.legend(loc='upper right', fontsize=9)
    
    def _create_initial_condition_control(self, parent, key, label, row=0, column=0):
        """Create a labeled initial condition entry"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky="ew", pady=2)
        
        # Create label
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        
        # Create variable and entry
        self.initial_vars[key] = tk.DoubleVar(value=self.model.initial_conditions[key])
        entry = ttk.Entry(frame, textvariable=self.initial_vars[key], width=8)
        entry.grid(row=0, column=1, sticky="e", padx=5)
        
        return frame
    
    def _export_data(self):
        """Export simulation results to CSV"""
        if self.model.last_results is None:
            messagebox.showinfo("No Data", "Run a simulation first")
            return
            
        try:
            # Get file path
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Export Simulation Data"
            )
            
            if not filename:
                return  # User cancelled
                
            # Get data
            t, solution = self.model.last_results
            
            # Create column names
            if self.model.model_config['age_structure']:
                columns = ['Time', 'S_y', 'I_y', 'R_y', 'S_e', 'I_e', 'R_e']
                if self.model.model_config['use_mutation'] and solution.shape[1] > 6:
                    columns.append('M')
            else:
                columns = ['Time', 'S', 'I', 'R']
                idx = 3
                
                if self.model.model_config['use_D']:
                    columns.append('D')
                    idx += 1
                    
                if self.model.model_config['use_V']:
                    columns.append('V')
                    idx += 1
                    
                if self.model.model_config['use_Q']:
                    columns.append('Q')
                    idx += 1
                    
                if self.model.model_config['use_mutation'] and idx < solution.shape[1]:
                    columns.append('M')
            
            # Create DataFrame
            df = pd.DataFrame(data=np.column_stack((t, solution)), columns=columns)
            
            # Export to CSV
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Complete", f"Data exported to {filename}")
            
        except Exception as e:
            self._show_error(f"Error exporting data: {str(e)}")
    
    def _export_plot(self):
        """Export current plot as image"""
        if self.model.last_results is None:
            messagebox.showinfo("No Plot", "Run a simulation first")
            return
            
        try:
            # Get file path
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG Files", "*.png"), 
                    ("JPEG Files", "*.jpg"),
                    ("PDF Files", "*.pdf"),
                    ("SVG Files", "*.svg"),
                    ("All Files", "*.*")
                ],
                title="Export Plot"
            )
            
            if not filename:
                return  # User cancelled
                
            # Save figure
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            
            messagebox.showinfo("Export Complete", f"Plot exported to {filename}")
            
        except Exception as e:
            self._show_error(f"Error exporting plot: {str(e)}")

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = SuperModelGUI(root)
    root.mainloop()
