# 2.3.1 Import delle librerie e definizione dei parametri globali
# Import delle librerie
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import atan2, degrees
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# Parametri Globali
GOAL_X = 120  # coordinata x della porta
GOAL_Y = 40  # coordinata y centrale della porta
MATCH_IDS = ['3869685', '3857256', '3857255', '3857261', '3857274', '3857286',
             '3857279', '3869151', '3857287', '3857266', '3857273', '3869552', '3869321']
MATCH_SELEZIONATO = '3869685'
SECOND_MATCH = '3869321'
EVENTS_DIR = os.path.join('open-data', 'data', 'events')

HOME_COLOR_FIRST = "skyblue"
AWAY_COLOR_FIRST = "darkblue"
HOME_COLOR_SECOND = "skyblue"
AWAY_COLOR_SECOND = "darkorange"


# 2.3.2 Caricamento ed estrazione dei dati
# Caricamento degli eventi
def load_events(match_ids, events_dir):
    all_events = []
    for mid in match_ids:
        file_path = os.path.join(events_dir, f'{mid}.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            events = json.load(f)
        for event in events:
            event['match_id'] = mid
        all_events.extend(events)
    return all_events


# Estrazione degli eventi di tipo "Shot"
def extract_shots(events):
    shots = [event for event in events if event.get('type', {}).get('name') == 'Shot']
    data = []
    for event in shots:
        shot = {
            'match_id': event.get('match_id'),
            'team': event.get('team', {}).get('name'),
            'player': event.get('player', {}).get('name'),
            'minute': event.get('minute'),
            'location': event.get('location'),
            'outcome': event.get('shot', {}).get('outcome', {}).get('name') if event.get('shot', {}).get(
                'outcome') else None,
            'technique': event.get('shot', {}).get('technique', {}).get('name') if event.get('shot', {}).get(
                'technique') else 'Unknown',
            'end_location': event.get('shot', {}).get('end_location'),
            'shot_type': event.get('shot', {}).get('type', {}).get('name')
        }
        data.append(shot)
    return pd.DataFrame(data)


# Estrazione degli eventi di tipo "Goal Keeper"
def extract_goalkeepers(events):
    goalkeeper_events = [event for event in events if event.get('type', {}).get('name') == 'Goal Keeper']
    data = []
    for event in goalkeeper_events:
        keeper_info = event.get('goalkeeper', {})
        data.append({
            'match_id': event.get('match_id'),
            'minute': event.get('minute'),
            'team': event.get('team', {}).get('name'),
            'goalkeeper': event.get('player', {}).get('name'),
            'save_outcome': keeper_info.get('outcome', {}).get('name') if keeper_info.get('outcome') else None,
            'technique': keeper_info.get('technique', {}).get('name') if keeper_info.get('technique') else 'Unknown'
        })
    return pd.DataFrame(data)


# 2.3.3 Feature Engineering e Normalizzazione
# 2.3.3.1 Pre-processing dei tiri
def preprocess_shots(df):
    df[['x', 'y']] = pd.DataFrame(df['location'].tolist(), index=df.index)
    df.drop(columns=['location'], inplace=True)
    df['end_z'] = df['end_location'].apply(lambda loc: loc[2] if isinstance(loc, list) and len(loc) > 2 else np.nan)
    df['end_x'] = df['end_location'].apply(lambda loc: loc[0] if isinstance(loc, list) and len(loc) > 0 else np.nan)
    df['end_y'] = df['end_location'].apply(lambda loc: loc[1] if isinstance(loc, list) and len(loc) > 1 else np.nan)
    df.drop(columns=['end_location'], inplace=True)
    df['technique'] = df['technique'].fillna('Unknown')
    df.dropna(subset=['x', 'y'], inplace=True)
    df['distance'] = np.sqrt((GOAL_X - df['x']) ** 2 + (GOAL_Y - df['y']) ** 2)

    def compute_angle(row, goal_width=8):
        dx = GOAL_X - row['x']
        left_post = GOAL_Y - goal_width / 2
        right_post = GOAL_Y + goal_width / 2
        angle_left = atan2((left_post - row['y']), dx)
        angle_right = atan2((right_post - row['y']), dx)
        return abs(degrees(angle_right - angle_left))

    df['angle'] = df.apply(compute_angle, axis=1)
    df['goal'] = df['outcome'].apply(lambda x: 1 if x == 'Goal' else 0)
    scaler = StandardScaler()
    df[['distance_norm', 'angle_norm']] = scaler.fit_transform(df[['distance', 'angle']])
    return df


# 2.3.3.2 Associazione del tiro al portiere avversario
def assign_goalkeeper(df, df_goalkeepers, match_selected):
    df_match = df[df['match_id'] == match_selected].copy()
    df_gk = df_goalkeepers[df_goalkeepers['match_id'] == match_selected].copy()

    gk_dict = df_gk.set_index('team')['goalkeeper'].to_dict()

    teams = set(df_match['team'].unique())

    def get_opponent_goalkeeper(team):
        opponent = (teams - {team}).pop() if len(teams) > 1 else team
        return gk_dict.get(opponent, 'Unknown')

    df_match['goalkeeper'] = df_match['team'].apply(get_opponent_goalkeeper)
    return df_match


# 2.4 Training dei modelli per calcolare gli Expected Goals (xG)
def train_model(df, model, col_name, match_selected=MATCH_SELEZIONATO):
    """
    Addestramento del modello per calcolare xG:
      - Preparazione delle feature (normalizzate e variabili dummy per la tecnica)
      - Divisione dei dati in training e test (escludendo i match selezionati dal training)
      - Bilanciamento del training set usando SMOTE
      - Addestramento del modello e calcola xG
      - Calcolo delle metriche sul test set
    """
    df_tech = pd.get_dummies(df['technique'], prefix='tech')
    X = pd.concat([df[['distance_norm', 'angle_norm']], df_tech], axis=1)
    y = df['goal']
    train_idx = df['match_id'] != match_selected
    test_idx = df['match_id'] == match_selected
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    model.fit(X_train_bal, y_train_bal)
    df[col_name] = model.predict_proba(X)[:, 1]
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    best_threshold = 0.8
    y_pred_class = (y_pred_proba >= best_threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_class),
        'precision': precision_score(y_test, y_pred_class, zero_division=0),
        'recall': recall_score(y_test, y_pred_class, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_class, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_class),
        'classification_report': classification_report(y_test, y_pred_class, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_threshold': best_threshold
    }
    return model, df, X, metrics, X_test, y_test, y_pred_proba

def print_model_metrics(model_name, metrics):
    print(f"\nMetriche sul test set - {model_name}:")
    for key, value in metrics.items():
        if key != 'classification_report':
            print(f"{key}: {value}")
    print(metrics['classification_report'])


def plot_roc_curve(y_test, y_pred_proba, model_name="Modello"):
    """
    Grafico della curva ROC.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Tasso di Falsi Positivi')
    plt.ylabel('Tasso di Veri Positivi')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# 2.4.1 Data visualization dei tiri
def draw_pitch(ax=None, pitch_length=120, pitch_width=80, pitch_color='white', line_color='black'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor(pitch_color)
    pitch = patches.Rectangle((0, 0), pitch_length, pitch_width, linewidth=2, edgecolor=line_color, facecolor='none')
    ax.add_patch(pitch)
    ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color=line_color, linewidth=2)
    center_circle = patches.Circle((pitch_length / 2, pitch_width / 2), radius=9.15, linewidth=2, edgecolor=line_color,
                                   facecolor='none')
    ax.add_patch(center_circle)
    ax.plot(pitch_length / 2, pitch_width / 2, marker='o', markersize=2, color=line_color)
    left_penalty = patches.Rectangle((0, (pitch_width - 42.38) / 2), 18.62, 42.38, linewidth=2, edgecolor=line_color,
                                     facecolor='none')
    ax.add_patch(left_penalty)
    right_penalty = patches.Rectangle((pitch_length - 18.82, (pitch_width - 42.38) / 2), 18.82, 42.38, linewidth=2,
                                      edgecolor=line_color, facecolor='none')
    ax.add_patch(right_penalty)
    left_goal = patches.Rectangle((0, (pitch_width - 18.32) / 2), 5.5, 18.32, linewidth=2, edgecolor=line_color,
                                  facecolor='none')
    ax.add_patch(left_goal)
    right_goal = patches.Rectangle((pitch_length - 5.5, (pitch_width - 18.32) / 2), 5.5, 18.32, linewidth=2,
                                   edgecolor=line_color, facecolor='none')
    ax.add_patch(right_goal)
    ax.plot(11, pitch_width / 2, marker='o', markersize=2, color=line_color)
    ax.plot(pitch_length - 11, pitch_width / 2, marker='o', markersize=2, color=line_color)
    left_arc = patches.Arc((11, pitch_width / 2), height=18.3, width=22, angle=0, theta1=320, theta2=40,
                           color=line_color, linewidth=2)
    ax.add_patch(left_arc)
    right_arc = patches.Arc((pitch_length - 11, pitch_width / 2), height=18.3, width=22, angle=0, theta1=140,
                            theta2=220, color=line_color, linewidth=2)
    ax.add_patch(right_arc)
    ax.set_xlim(-5, pitch_length + 5)
    ax.set_ylim(-5, pitch_width + 5)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax


def get_color_mapping(match_selected, teams):
    mapping = {}
    if match_selected == MATCH_SELEZIONATO:
        mapping[teams[0]] = HOME_COLOR_FIRST
        if len(teams) > 1:
            mapping[teams[1]] = AWAY_COLOR_FIRST
    elif match_selected == SECOND_MATCH:
        mapping[teams[0]] = HOME_COLOR_SECOND
        if len(teams) > 1:
            mapping[teams[1]] = AWAY_COLOR_SECOND
    else:
        mapping[teams[0]] = HOME_COLOR_FIRST
        if len(teams) > 1:
            mapping[teams[1]] = AWAY_COLOR_FIRST
    return mapping


def visualize_shot_positions(df, match_selected):
    df_viz = df[df['match_id'] == match_selected].copy()
    teams = df_viz['team'].unique()
    color_mapping = get_color_mapping(match_selected, teams)
    n = len(teams)
    fig, axs = plt.subplots(1, n, figsize=(8 * n, 8), sharey=True)
    if n == 1:
        axs = [axs]
    for ax, team in zip(axs, teams):
        draw_pitch(ax)
        team_data = df_viz[df_viz['team'] == team]
        no_goals = team_data[team_data['goal'] == 0]
        goals = team_data[team_data['goal'] == 1]
        team_color = color_mapping.get(team, "black")
        ax.scatter(no_goals['x'], no_goals['y'], color=team_color, marker='o', label="No Goal", alpha=0.7)
        ax.scatter(goals['x'], goals['y'], color=team_color, marker='*', s=100, label="Goal", alpha=0.9)
        median_x = team_data['x'].median()
        ax.set_xlim(60, 120) if median_x > 60 else ax.set_xlim(0, 60)
        ax.set_title(f"Tiri per {team}")
        ax.legend(loc='lower right')
    plt.suptitle("Posizioni di tiro")
    plt.show()


def visualize_final_shots_distribution(df, match_selected):
    df_viz = df[df['match_id'] == match_selected].copy()
    teams = df_viz['team'].unique()
    bins = np.linspace(30, 50, 21)
    mapping = get_color_mapping(match_selected, teams)
    fig, axes = plt.subplots(1, len(teams), figsize=(15, 6), sharey=True)
    for ax, team in zip(axes, teams):
        team_shots = df_viz[(df_viz['team'] == team) & (df_viz['end_y'].notna())]
        ax.hist(team_shots['end_y'], bins=bins, alpha=0.7,
                label=team, color=mapping.get(team, "darkblue"))
        left_post = GOAL_Y - 8 / 2
        right_post = GOAL_Y + 8 / 2
        ax.axvline(x=left_post, color='black', linestyle='--', label='Palo sinistro')
        ax.axvline(x=right_post, color='black', linestyle='--', label='Palo destro')
        ax.set_xlabel("Linea di fondocampo")
        ax.set_title(f"Distribuzione dei tiri finali - {team}")
        ax.legend()
    axes[0].set_ylabel("Frequenza")
    plt.suptitle("Distribuzione dei tiri lungo la porta", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def visualize_heatmap(df, match_selected):
    df_viz = df[df['match_id'] == match_selected].copy()
    goal_shots = df_viz[(df_viz['end_y'].notna()) & (df_viz['end_z'].notna())].copy()
    goal_shots = goal_shots[goal_shots['end_x'] >= 115]
    door_width = 8.0
    door_height = 2.67
    left_post_pitch = GOAL_Y - door_width / 2
    goal_shots['goal_y_trans'] = goal_shots['end_y'] - left_post_pitch
    y_bins = np.linspace(0, door_width, 31)
    z_bins = np.linspace(0, door_height, 25)
    heatmap, _, _ = np.histogram2d(goal_shots['goal_y_trans'], goal_shots['end_z'], bins=[y_bins, z_bins])
    heatmap = np.flipud(np.rot90(heatmap))
    fig, ax = plt.subplots(figsize=(8, 6))
    goal_rect = patches.Rectangle((0, 0), door_width, door_height, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(goal_rect)
    im = ax.imshow(heatmap, extent=[0, door_width, 0, door_height], cmap='hot', alpha=0.6, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, label="Numero di tiri", shrink=0.5)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlim(0, door_width)
    ax.set_ylim(0, door_height)
    ax.set_xlabel("Larghezza porta (m)")
    ax.set_ylabel("Altezza porta (m)")
    ax.set_title("Heatmap dei tiri in porta")
    teams_unique = goal_shots['team'].unique()
    mapping2 = get_color_mapping(match_selected, teams_unique)
    for idx, shot in goal_shots.iterrows():
        x_val = shot['goal_y_trans']
        z_val = shot['end_z']
        team = shot['team']
        shot_type = shot.get('shot_type')
        if shot['goal'] == 1:
            marker = "D" if shot_type == "Penalty" else "*"
            label = f'Gol {team} (Penalty)' if shot_type == "Penalty" else f'Gol {team} (Non-Penalty)'
            color = mapping2.get(team, 'black')
            ax.scatter(x_val, z_val, marker=marker, s=100, color=color, edgecolor='black', label=label)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='lower left', bbox_to_anchor=(1.05, 1),
              borderaxespad=0.)
    ax.set_aspect('equal')
    plt.show()

# 2.5 Gli Expected Goals Saved (xGS)
def compute_goalkeeper_aggregates(df, df_goalkeepers, match_selected):
    df_viz = assign_goalkeeper(df, df_goalkeepers, match_selected).copy()
    aggregated = df_viz.groupby('goalkeeper', as_index=False).agg({'xG': 'sum', 'goal': 'sum', 'team': 'first'})
    aggregated.rename(columns={'goal': 'goals_conceded'}, inplace=True)
    aggregated['xGS'] = aggregated['xG'] - aggregated['goals_conceded']
    shots_faced = df_viz.groupby('goalkeeper').size().rename("shots_faced").reset_index()
    gk_stats = pd.merge(aggregated, shots_faced, on='goalkeeper')
    gk_stats['saves'] = gk_stats['shots_faced'] - gk_stats['goals_conceded']
    gk_stats['save_percentage'] = gk_stats['saves'] / gk_stats['shots_faced']
    return df_viz, gk_stats


def get_goalkeeper_row_data(df, df_goalkeepers, match_selected):
    df_viz = assign_goalkeeper(df, df_goalkeepers, match_selected).copy()
    df_viz.sort_values('minute', inplace=True)
    return df_viz

# 2.5.1 Data visualization degli xGS per i portieri
def invert_mapping(mapping, match_selected):
    if match_selected == MATCH_SELEZIONATO:
        inversion = {
            "darkblue": "skyblue",
            "skyblue": "darkblue"
        }
    elif match_selected == SECOND_MATCH:
        inversion = {
            "skyblue": "darkorange",
            "darkorange": "skyblue"
        }
    else:
        inversion = {}
    return {team: inversion.get(color, color) for team, color in mapping.items()}


def visualize_goalkeeper_stats(df, df_goalkeepers, match_selected):
    df_viz, gk_perf = compute_goalkeeper_aggregates(df, df_goalkeepers, match_selected)
    teams = list(df_viz['team'].unique())
    mapping = get_color_mapping(match_selected, teams)
    mapping = invert_mapping(mapping, match_selected)

    print("Expected Goal Saved (xGS) per portiere:")
    print(gk_perf[['goalkeeper', 'xG', 'goals_conceded', 'xGS']])

    fig, ax = plt.subplots(figsize=(8, 6))
    index = np.arange(len(gk_perf))
    bar_width = 0.35
    colors_xG = ["red" for _ in gk_perf['goalkeeper']]
    colors_goals = [mapping.get(team, "blue") for team in gk_perf['team']]
    ax.bar(index, gk_perf['xG'], bar_width, alpha=0.8, color=colors_xG, label='xG Conceded')
    ax.bar(index + bar_width, gk_perf['goals_conceded'], bar_width, alpha=0.8, color=colors_goals,
           label='Goals Conceded')
    plt.xlabel('Portiere')
    plt.ylabel('Valore')
    plt.title('xG e Goal Subiti per Portiere')
    plt.xticks(index + bar_width / 2, gk_perf['goalkeeper'])
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, row in gk_perf.iterrows():
        ax.scatter(row['xG'], row['goals_conceded'], s=100, color=mapping.get(row['team'], "blue"),
                   label=row['goalkeeper'])
        ax.text(row['xG'] * 1.01, row['goals_conceded'] * 1.01, row['goalkeeper'], fontsize=9)
    max_val = max(gk_perf['xG'].max(), gk_perf['goals_conceded'].max())
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--')
    ax.set_xlabel('xG Subiti')
    ax.set_ylabel('Goal Subiti')
    ax.set_title('Confronto: xG vs Goal Subiti')
    ax.legend()
    plt.show()

    print("Statistiche di salvataggi per portiere:")
    print(gk_perf[['goalkeeper', 'shots_faced', 'goals_conceded', 'saves', 'save_percentage']])
    colors_save = [mapping.get(team, "purple") for team in gk_perf['team']]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(gk_perf['goalkeeper'], gk_perf['save_percentage'], color=colors_save, alpha=0.7)
    ax.set_xlabel('Portiere')
    ax.set_ylabel('Percentuale di salvataggio')
    ax.set_title('Percentuale di salvataggio per Portiere')
    plt.ylim(0, 1)
    plt.show()


def visualize_goalkeeper_xgs(df, df_goalkeepers, match_selected):
    _, gk_stats = compute_goalkeeper_aggregates(df, df_goalkeepers, match_selected)
    teams = list(gk_stats['team'].unique())
    mapping = get_color_mapping(match_selected, teams)

    if match_selected != MATCH_SELEZIONATO:
        mapping = invert_mapping(mapping, match_selected)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [mapping.get(team, "blue") for team in gk_stats['team']]
    ax.bar(gk_stats['goalkeeper'], gk_stats['xGS'], color=colors, alpha=0.7)
    ax.set_xlabel('Portiere')
    ax.set_ylabel('xGS (Expected Goals Saved)')
    ax.set_title('Confronto xGS per Portiere')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_goalkeeper_time_series(df, df_goalkeepers, match_selected):
    df_viz = get_goalkeeper_row_data(df, df_goalkeepers, match_selected)
    teams = list(df_viz['team'].unique())
    mapping = get_color_mapping(match_selected, teams)
    mapping = invert_mapping(mapping, match_selected)  # Inversione dei colori

    plt.figure(figsize=(10, 6))
    for gk in df_viz['goalkeeper'].unique():
        gk_shots = df_viz[df_viz['goalkeeper'] == gk].copy()
        gk_shots['cumulative_xG'] = gk_shots['xG'].cumsum()
        team = gk_shots['team'].iloc[0]
        plt.plot(gk_shots['minute'], gk_shots['cumulative_xG'], marker='o',
                 color=mapping.get(team, "blue"), label=gk)
    plt.xlabel('Minuto di Gioco')
    plt.ylabel('xG Cumulati')
    plt.title('Andamento xG nel Tempo')
    plt.legend()
    plt.show()


def visualize_goalkeeper_xgs_over_time(df, df_goalkeepers, match_selected):
    """
    Evoluzione temporale della xGS (Expected Goals Saved) per ciascun portiere.
    """
    df_viz = get_goalkeeper_row_data(df, df_goalkeepers, match_selected)
    df_viz['xGS_current'] = df_viz['xG'] - df_viz['goal']
    df_viz['cumulative_xGS'] = df_viz.groupby('goalkeeper')['xGS_current'].cumsum()
    teams = list(df_viz['team'].unique())
    mapping = get_color_mapping(match_selected, teams)
    mapping = invert_mapping(mapping, match_selected)  # Inversione dei colori

    fig, ax = plt.subplots(figsize=(10, 6))
    for gk in df_viz['goalkeeper'].unique():
        gk_data = df_viz[df_viz['goalkeeper'] == gk]
        team = gk_data['team'].iloc[0]
        ax.plot(gk_data['minute'], gk_data['cumulative_xGS'], marker='o',
                color=mapping.get(team, "blue"), label=gk)
    ax.set_xlabel('Minuto di Gioco')
    ax.set_ylabel('xGS Cumulati')
    ax.set_title('Andamento xGS nel Tempo')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.legend()
    plt.show()

def visualize_match(match_id, df_shots, df_goalkeepers):
    visualize_shot_positions(df_shots, match_id)
    visualize_final_shots_distribution(df_shots, match_id)
    visualize_heatmap(df_shots, match_id)
    visualize_goalkeeper_stats(df_shots, df_goalkeepers, match_id)
    visualize_goalkeeper_xgs(df_shots, df_goalkeepers, match_id)
    visualize_goalkeeper_time_series(df_shots, df_goalkeepers, match_id)
    visualize_goalkeeper_xgs_over_time(df_shots, df_goalkeepers, match_id)


# 2.6 Confronto tra i portieri nelle partite
def comparative_goalkeeper_analysis(df_shots, df_goalkeepers, match1, match2):
    _, gk_stats1 = compute_goalkeeper_aggregates(df_shots, df_goalkeepers, match1)
    _, gk_stats2 = compute_goalkeeper_aggregates(df_shots, df_goalkeepers, match2)

    gk_stats1['match'] = match1
    gk_stats2['match'] = match2

    combined_gk = pd.concat([gk_stats1, gk_stats2], ignore_index=True)

    color_mapping = {match1: "gold", match2: "silver"}
    marker_mapping = {match1: "*", match2: "s"}

    nicknames = {
        "Damián Emiliano Martínez": "ARG",
        "Hugo Lloris": "FRA",
        "Andries Noppert": "NET"
    }

    # Plot comparativo: xG subiti (xG) vs Goal subiti (goals_conceded)
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, row in combined_gk.iterrows():
        m = row['match']
        goalkeeper_name = row['goalkeeper']
        # Recupera il soprannome se presente, altrimenti usa il nome originale
        nickname = nicknames.get(goalkeeper_name, goalkeeper_name)
        ax.scatter(row['xG'], row['goals_conceded'],
                   s=100,
                   color=color_mapping.get(m, "blue"),
                   marker=marker_mapping.get(m, "o"),
                   label=nickname)
        ax.text(row['xG'] * 1.01, row['goals_conceded'] * 1.01, nickname, fontsize=9)

    max_val = max(combined_gk['xG'].max(), combined_gk['goals_conceded'].max())
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--')
    ax.set_xlabel('xG Subiti')
    ax.set_ylabel('Goal Subiti')
    ax.set_title('Confronto: xG vs Goal Subiti per Portiere nei due match')
    ax.legend(loc='best')
    plt.show()


def comparative_goalkeeper_xgs_over_time(df_shots, df_goalkeepers, match1, match2):
    df_viz1 = get_goalkeeper_row_data(df_shots, df_goalkeepers, match1)
    df_viz1['xGS_current'] = df_viz1['xG'] - df_viz1['goal']
    df_viz1['cumulative_xGS'] = df_viz1.groupby('goalkeeper')['xGS_current'].cumsum()
    df_viz1['match'] = match1

    df_viz2 = get_goalkeeper_row_data(df_shots, df_goalkeepers, match2)
    df_viz2['xGS_current'] = df_viz2['xG'] - df_viz2['goal']
    df_viz2['cumulative_xGS'] = df_viz2.groupby('goalkeeper')['xGS_current'].cumsum()
    df_viz2['match'] = match2

    combined_viz = pd.concat([df_viz1, df_viz2], ignore_index=True)

    teams = list(combined_viz['team'].unique())
    color_mapping_match1 = invert_mapping(get_color_mapping(match1, teams), match1)
    color_mapping_match2 = invert_mapping(get_color_mapping(match2, teams), match2)

    fig, ax = plt.subplots(figsize=(10, 6))

    for match in [match1, match2]:
        data_match = combined_viz[combined_viz['match'] == match]
        for gk in data_match['goalkeeper'].unique():
            gk_data = data_match[data_match['goalkeeper'] == gk]
            team = gk_data['team'].iloc[0]
            color = color_mapping_match1.get(team, "steelblue") if match == match1 else color_mapping_match2.get(team,
                                                                                                            "steelblue")
            ax.plot(gk_data['minute'], gk_data['cumulative_xGS'], marker='o',
                    color=color, label=f"{gk}")

    ax.set_xlabel('Minuto di Gioco')
    ax.set_ylabel('xGS Cumulati')
    ax.set_title('Confronto xGS nel Tempo tra i Portieri nei due match')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.legend()
    plt.show()

# 2.7 Funzione Main
def main():
    # Caricamento ed estrazione dei dati
    events = load_events(MATCH_IDS, EVENTS_DIR)
    print(f"Numero totale di eventi caricati: {len(events)}")
    df_shots = extract_shots(events)
    print(f"Numero di tiri estratti: {len(df_shots)}")
    df_goalkeepers = extract_goalkeepers(events)
    print(f"Numero di eventi portieri: {len(df_goalkeepers)}")
    df_shots = preprocess_shots(df_shots)
    print("Prime righe dei tiri preprocessati:")
    print(df_shots.head())

    # Training dei modelli
    modelli = []
    for nome, modello, col in [
        ("Regressione Logistica", LogisticRegression(C=1.0, solver='liblinear', random_state=42), "xG_logreg"),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), "xG_rf"),
        ("MLP", MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42), "xG_mlp")
    ]:
        modello_trained, df_shots, _, metrics, X_test, y_test, y_pred_proba = train_model(df_shots, modello, col)
        modelli.append({
            "nome": nome,
            "metrics": metrics,
            "y_test": y_test,
            "y_pred_proba": y_pred_proba,
            "xG": df_shots[col]
        })
        print_model_metrics(nome, metrics)

    # Selezione del modello migliore in base al roc_auc
    best_model = max(modelli, key=lambda m: m['metrics']['roc_auc'])
    best_model_name = best_model["nome"]
    best_metrics = best_model["metrics"]
    best_y_test = best_model["y_test"]
    best_y_pred_proba = best_model["y_pred_proba"]
    df_shots['xG'] = best_model["xG"]

    print(f"\nIl modello migliore è: {best_model_name}")
    print(f"ROC AUC: {best_metrics['roc_auc']}, F1 Score: {best_metrics['f1_score']}")
    plot_roc_curve(best_y_test, best_y_pred_proba, model_name=best_model_name)

    # Visualizzazione per i match di interesse
    print(f"\nVisualizzazioni per il match: {MATCH_SELEZIONATO}")
    visualize_match(MATCH_SELEZIONATO, df_shots, df_goalkeepers)
    print(f"\nVisualizzazioni per il match: {SECOND_MATCH}")
    visualize_match(SECOND_MATCH, df_shots, df_goalkeepers)

    # Analisi comparativa aggregata per i portieri tra i due match
    comparative_goalkeeper_analysis(df_shots, df_goalkeepers, MATCH_SELEZIONATO, SECOND_MATCH)
    comparative_goalkeeper_xgs_over_time(df_shots, df_goalkeepers, MATCH_SELEZIONATO, SECOND_MATCH)
if __name__ == "__main__":
    main()
