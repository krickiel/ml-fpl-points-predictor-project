# fpl-points-predictor
The goal of this project is to predict Fantasy Premier League player points for the remaining gameweeks (GW21-GW38) using supervised learning techniques and dynamic simulation.

## Methodology

The pipeline consists of four main stages implemented in `fpl_project.py`:

1.  **Data Preprocessing & Feature Engineering:**
    * Merged historical match data with static player statistics.
    * **Custom Features:** Created `relative_difficulty` metric (difference between own team strength and opponent strength) and rolling averages (`form_last_3`) to capture temporal trends.
    * **Domain Knowledge:** Implemented specific scaling for team strength (1-10 scale) and position encoding to handle defensive points logic.

2.  **Model Selection:**
    * The script automatically trains and compares three models: **Linear Regression**, **KNN**, and **Random Forest Regressor**.
    * The best performing model (selected based on $R^2$ score on the test set) is used for the final simulation.

3.  **Dynamic Simulation (Feedback Loop):**
    * Instead of static predictions, the system runs a sequential simulation for the remaining fixtures.
    * Predicted points for GW $n$ are fed back into the history to update the player's form for GW $n+1$. This prevents static outputs and simulates performance momentum.

## Repository Structure

* `fpl_project.py` - Main script containing the ETL process, model training, and simulation logic.
* `fpl_detailed_history_2025.csv` - Historical training data (Gameweeks 1-20).
* `team_match_stats.csv` - Fixture schedule for the remainder of the season.
* `fpl_player_statistics.csv` - Player metadata (cost, position, team mapping).
* `fpl_ultimate.xlsx` - Output file containing xP (Expected Points) projections.
