import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load data
try:
    df_history = pd.read_csv('fpl_detailed_history_2025.csv')
    df_static = pd.read_csv('fpl_player_statistics.csv')
except FileNotFoundError:
    print("error missing csv")
    exit()

# map club names
team_name_to_id = {
    'Arsenal': 1, 'Aston Villa': 2, 'Bournemouth': 3, 'Brentford': 4,
    'Brighton': 5, 'Burnley': 6, 'Chelsea': 7, 'Crystal Palace': 8,
    'Everton': 9, 'Fulham': 10, 'Leeds': 11, 'Leeds United': 11,
    'Liverpool': 12, 'Man City': 13, 'Manchester City': 13,
    'Man Utd': 14, 'Manchester Utd': 14, 'Manchester United': 14,
    'Newcastle': 15, 'Newcastle Utd': 15,
    'Nottingham': 16, 'Nottingham Forest': 16, "Nott'ham Forest": 16,
    'Sunderland': 17, 'Tottenham': 18, 'West Ham': 19, 'Wolves': 20
}

if 'team' not in df_static.columns:
    print("adding team id column...")
    df_static['team'] = df_static['club_name'].map(team_name_to_id)
    df_static['team'] = df_static['team'].fillna(0).astype(int)

# map positions
position_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4,
                'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}

if 'position_name' in df_static.columns:
    df_static['position_id'] = df_static['position_name'].map(
        position_map).fillna(3).astype(int)

# merge and clean
static_info = df_static[['player_name', 'position_name',
                         'position_id', 'club_name', 'now_cost', 'team']]
df = pd.merge(df_history, static_info, on='player_name', how='left')
df = df[df['minutes'] > 0].copy()
df = df.sort_values(['player_name', 'round'])

# team strength 1-10
base_team_strength = {
    1: 10,  # arsenal
    2: 8,   # aston villa
    3: 5,   # bournemouth
    4: 5,   # brentford
    5: 5,   # brighton
    6: 1,   # burnley
    7: 7,   # chelsea
    8: 5,   # crystal palace
    9: 5,   # everton
    10: 3,  # fulham
    11: 5,  # leeds
    12: 8,  # liverpool
    13: 10,  # man city
    14: 6,  # man utd
    15: 6,  # newcastle
    16: 3,  # nottingham
    17: 5,  # sunderland
    18: 5,  # tottenham
    19: 1,  # west ham
    20: 1   # wolves
}


def calculate_difficulty(row):
    opponent_id = row['opponent_team']
    is_home = row['was_home']
    base = base_team_strength.get(opponent_id, 5)
    # +2 for away games
    return base if is_home else base + 2


if 'opponent_team' in df.columns:
    df['difficulty'] = df.apply(calculate_difficulty, axis=1)
    if 'was_home' in df.columns:
        df['is_home'] = df['was_home'].astype(int)

# features


def create_features(group):
    # target variable
    group['target_points'] = group['total_points'].shift(-1)

    # core stats
    group['form_last_3'] = group['total_points'].rolling(
        window=3, min_periods=1).mean().shift(1)
    group['points_last_match'] = group['total_points'].shift(1)
    group['season_avg'] = group['total_points'].expanding().mean().shift(1)
    group['minutes_avg'] = group['minutes'].expanding().mean().shift(1)
    group['total_points_so_far'] = group['total_points'].cumsum().shift(1)

    # defense stats
    if 'clean_sheets' in group.columns:
        group['clean_sheets_last_3'] = group['clean_sheets'].rolling(
            window=3, min_periods=1).mean().shift(1)
    if 'goals_conceded' in group.columns:
        group['goals_conceded_last_3'] = group['goals_conceded'].rolling(
            window=3, min_periods=1).mean().shift(1)

    # relative difficulty
    my_team_id = group['team'].iloc[0]
    my_team_strength = base_team_strength.get(my_team_id, 5)

    if 'difficulty' in group.columns:
        group['next_match_difficulty'] = group['difficulty'].shift(-1)
        group['relative_difficulty'] = group['next_match_difficulty'] - \
            my_team_strength

    if 'is_home' in group.columns:
        group['next_is_home'] = group['is_home'].shift(-1)

    return group


df_ml = df.groupby('player_name', group_keys=False).apply(
    create_features, include_groups=False)
df_ml = df_ml.dropna(subset=[
                     'target_points', 'next_match_difficulty', 'season_avg', 'relative_difficulty'])

# select features
potential_features = [
    'form_last_3', 'points_last_match', 'season_avg', 'minutes_avg',
    'total_points_so_far', 'now_cost',
    'next_match_difficulty', 'relative_difficulty', 'next_is_home',
    'clean_sheets_last_3', 'goals_conceded_last_3', 'position_id'
]
features = [f for f in potential_features if f in df_ml.columns]

for col in features:
    df_ml[col] = df_ml[col].fillna(0)

# train test split
target = 'target_points'
X = df_ml[features]
y = df_ml[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# model comparison
print("\n training and comparing models")

models = {
    "Linear Regression": LinearRegression(),
    "KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
}

best_model = None
best_score = -999
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # using r2 score to compare
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"{name}: R2 = {score:.4f}, MAE = {mae:.4f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

print(f"\n best: {best_name} (using for simulation)")
final_model = best_model

print(f"model trained. features count: {len(features)}")
final_model = model

# current season stats
df['season_avg_current'] = df.groupby(
    'player_name')['total_points'].transform(lambda x: x.expanding().mean())
df['minutes_avg_current'] = df.groupby(
    'player_name')['minutes'].transform(lambda x: x.expanding().mean())
df['total_points_current'] = df.groupby(
    'player_name')['total_points'].transform(lambda x: x.cumsum())

latest_stats = df.sort_values('round').groupby('player_name').tail(1).copy()

# fill latest features
latest_stats['form_last_3'] = latest_stats['total_points'].rolling(
    window=3, min_periods=1).mean()
latest_stats['points_last_match'] = latest_stats['total_points']
latest_stats['season_avg'] = latest_stats['season_avg_current']
latest_stats['minutes_avg'] = latest_stats['minutes_avg_current']
latest_stats['total_points_so_far'] = latest_stats['total_points_current']
if 'clean_sheets' in df.columns:
    latest_stats['clean_sheets_last_3'] = latest_stats['clean_sheets'].rolling(
        window=3, min_periods=1).mean()
if 'goals_conceded' in df.columns:
    latest_stats['goals_conceded_last_3'] = latest_stats['goals_conceded'].rolling(
        window=3, min_periods=1).mean()

for col in features:
    if col not in latest_stats.columns:
        latest_stats[col] = 0
    latest_stats[col] = latest_stats[col].fillna(0)

# load schedule
try:
    df_schedule = pd.read_csv('team_match_stats.csv')
    schedule = {}
    for index, row in df_schedule.iterrows():
        try:
            gw = int(row['round'].split(' ')[1])
        except:
            continue
        if row['venue'] == 'Home':
            h_n, a_n = row['team'], row['opponent']
        else:
            h_n, a_n = row['opponent'], row['team']
        h_id, a_id = team_name_to_id.get(h_n), team_name_to_id.get(a_n)
        if h_id and a_id:
            if gw not in schedule:
                schedule[gw] = []
            if (h_id, a_id) not in schedule[gw]:
                schedule[gw].append((h_id, a_id))
    print(f"schedule loaded: {len(schedule)} gameweeks")
except:
    print("error loading schedule")
    schedule = {}

# dynamic simulation function
CURRENT_GW = 21
FINAL_GW = 38


def simulate_future_points(row):
    player_team = row['team']
    my_team_strength = base_team_strength.get(player_team, 5)

    # init simulation state
    simulated_history = [row['season_avg']] * 3
    current_form = row['form_last_3']
    if pd.isna(current_form):
        current_form = row['season_avg']

    points_accumulator = {1: 0, 3: 0, 5: 0, 10: 0, 'end': 0}

    # loop through weeks
    for gw in range(CURRENT_GW, FINAL_GW + 1):
        if gw not in schedule:
            continue
        matches = schedule[gw]

        opponent, is_home = None, None
        for home, away in matches:
            if player_team == home:
                opponent = away
                is_home = 1
                break
            elif player_team == away:
                opponent = home
                is_home = 0
                break

        if opponent is not None:
            # match params
            base_diff = base_team_strength.get(opponent, 5)
            diff = base_diff if is_home else base_diff + 2
            rel_diff = diff - my_team_strength

            # input
            input_row = {}
            for feat in features:
                if feat == 'form_last_3':
                    input_row[feat] = [current_form]  # using simulated form
                elif feat == 'next_match_difficulty':
                    input_row[feat] = [diff]
                elif feat == 'relative_difficulty':
                    input_row[feat] = [rel_diff]
                elif feat == 'next_is_home':
                    input_row[feat] = [is_home]
                else:
                    input_row[feat] = [row.get(feat, 0)]

            # predict
            input_data = pd.DataFrame(input_row)[features]
            pred_points = final_model.predict(input_data)[0]

            # update simulation history
            simulated_history.append(pred_points)
            if len(simulated_history) > 3:
                simulated_history.pop(0)

            # update form for next week
            new_recent_form = sum(simulated_history) / len(simulated_history)
            season_avg = row['season_avg']
            # mix form with season avg
            current_form = (new_recent_form * 0.7) + (season_avg * 0.3)

            # sum points
            w = gw - CURRENT_GW + 1
            if w == 1:
                points_accumulator[1] += pred_points
            if w <= 3:
                points_accumulator[3] += pred_points
            if w <= 5:
                points_accumulator[5] += pred_points
            if w <= 10:
                points_accumulator[10] += pred_points
            points_accumulator['end'] += pred_points

    return pd.Series([points_accumulator[1], points_accumulator[3], points_accumulator[5], points_accumulator[10], points_accumulator['end']])


if 'team' not in latest_stats.columns or latest_stats['team'].isnull().any():
    latest_stats['team'] = latest_stats['team'].fillna(0).astype(int)

print("running...")
latest_stats[['xP_1GW', 'xP_3GW', 'xP_5GW', 'xP_10GW', 'xP_EndSeason']
             ] = latest_stats.apply(simulate_future_points, axis=1)

# save results
final_table = latest_stats[['player_name', 'club_name', 'position_name', 'now_cost',
                            'xP_1GW', 'xP_3GW', 'xP_5GW', 'xP_10GW', 'xP_EndSeason']].copy()
final_table = final_table.sort_values(by='xP_EndSeason', ascending=False)
cols = ['xP_1GW', 'xP_3GW', 'xP_5GW', 'xP_10GW', 'xP_EndSeason']
final_table[cols] = final_table[cols].apply(lambda x: (x * 2).round() / 2)

print("\ntop players predicted:")
pd.options.display.float_format = '{:.1f}'.format
print(final_table.head(20))

final_table.to_excel('fpl_ultimate.xlsx', index=False)
print("excel: fpl_ultimate.xlsx")
