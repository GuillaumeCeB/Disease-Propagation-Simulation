# Disease Propagation Simulation

A powerful and flexible tool for simulating the spread of infectious diseases using various compartmental models.

## Features

- **Multiple epidemiological models**: SIR, SIRS, SIRD, SIRV, SIRQ with customizable parameters
- **Advanced dynamics**:
  - Waning immunity
  - Disease-induced mortality
  - Vaccination effects
  - Quarantine measures
  - Vital dynamics (births and deaths)
  - Seasonal variations
  - Population migration
  - Age-structured modeling
  - Virus mutation effects
- **Interactive GUI** with real-time visualization
- **Parameter customization** via sliders and input fields
- **Data export** capabilities for further analysis

## Installation

### Requirements
- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy
- Tkinter

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/disease-propagation-simulation.git
cd disease-propagation-simulation

# Install dependencies
pip install numpy pandas matplotlib scipy
```

## Usage

Run the simulation using:

```bash
python disease_propagation_simulation.py
```

The GUI will open, allowing you to:
1. Configure model parameters in the tabs
2. Run simulations with the "Run Simulation" button
3. View results in the interactive plot
4. Export data for further analysis

## Model Description

The simulation is based on compartmental models in epidemiology, with the following compartments:

- **S**: Susceptible individuals
- **I**: Infected individuals 
- **R**: Recovered individuals with immunity
- **D**: Deceased individuals (optional)
- **V**: Vaccinated individuals (optional)
- **Q**: Quarantined individuals (optional)

For age-structured models:
- **S_y, I_y, R_y**: Young population compartments
- **S_e, I_e, R_e**: Elderly population compartments

Key parameters:
- **β**: Transmission rate
- **γ**: Recovery rate
- **ξ**: Rate of waning immunity
- **α**: Disease-induced death rate
- **ρ**: Vaccination rate
- **κ**: Quarantine rate

## Extended Features

### Seasonal Variations
Model transmission rates that change with seasons using amplitude and phase shift parameters.

### Virus Mutation
Simulate pathogens that mutate over time, potentially increasing transmissibility.

### Migration Effects
Study the impact of population movement on disease spread.

### Age Structure
Analyze how diseases affect different age groups with varying transmission and recovery rates.

## Examples

The model can simulate various disease scenarios:

1. **Basic SIR model**: Simple epidemic progression
2. **Seasonal flu**: Yearly epidemic patterns with seasonality
3. **COVID-like scenario**: Higher mortality and quarantine measures
4. **Age-structured models**: Different impacts across age groups
5. **Mutation scenarios**: Evolution of pathogen over time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
