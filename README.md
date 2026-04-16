# Optimal Grocery Store Placements for Food Deserts in Boston
This project aims to use a genetic algorithm to suggest 10 placements for new grocery stores in the city of Boston to best serve the community. The specific goals are to priotize the maximum number of people served, low-income communities, and neighbordhoods with below average cardiac health, all while avoiding areas already served by a grocery store. This project is not concerned with smaller markets and bodegas, but rather large scale grocery stores that tend to be cheaper and have healthier food options than those previously mentioned. 

## Sample Output
The graphs below display the optimal grocery store placement based on fitness function after 100 generations. 
![Alt text]('/Users/annarose/Desktop/Screenshot 2026-04-15 at 3.39.05 PM.png')
![Alt text]('/Users/annarose/Desktop/Screenshot 2026-04-15 at 3.39.12 PM.png')


## Running the Code
### Step 1: Clone the repository
``` git clone https://github.com/anna-rose-19/CS4100-Project```

### Step 2: Install all requirements
This code uses python=3.13.9
```pip install -r requirements.txt```

### Step 3: Optionally change parameters
These values were decided upon for our results, however they can be altered for further experimentation
```N_STORES       = 10
POP_SIZE       = 50
N_GENERATIONS  = 60
ELITE_K        = 10
TOURNAMENT_K   = 4
MUTATION_RATE  = 0.7
MULTISWAP_PROB = 0.15
N_MULTISWAP    = 3
```
### Step 4: Run the genetic_algorithm.py file
Exit out of the original map pop ups to allow algorithm to continue.

### Step 5: View results and metrics
One the algorithm finishes running, a map of Boston will appear with 10 stores placed in locations that yielded high fitness scores. A graph displaying convergence will also appear. In the terminal, relevant metrics will print. 


## Future Direction
This project can be improved by considering: 
- Store proximity to public transit to allow for further accessibility
- Consideration of ethnic data in the store's reach to effectively serve the people there
- Updated datasets for more accuracy in health, income, and existing store data

## Collaborators
The individuals listed below contributed to the creation of this project and are available for further questions at the email addresses listed:
- Anna Rose (rose.anna@northeastern.edu)
- Anushka Poddar (poddar.an@northeastern.edu)
- Anjali Suresh (suresh.anj@northeastern.edu)
- Sean Yoo (yoo.sea@northeastern.edu)