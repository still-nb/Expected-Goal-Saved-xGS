# Expected-Goal-Saved-xGS

**Descrizione del Progetto**

Questo progetto si concentra sull'analisi degli Expected Goals Saved (xGS), una metrica avanzata per valutare la prestazione dei portieri nel calcio. L'analisi viene condotta utilizzando il dataset StatsBomb Open Data, ed è implementata in Python con l'uso di librerie di data science e machine learning.

Il codice fornisce un'analisi completa degli xGS attraverso l'aggregazione dei dati, la visualizzazione e la valutazione dei portieri in base ai gol subiti rispetto agli xG concessi.

**Requisiti**

Per eseguire lo script, è necessario avere installati i seguenti pacchetti:
- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- statsmodels
- imbalanced-learn

Puoi installarli eseguendo:

pip install pandas numpy scikit-learn matplotlib seaborn statsmodels imblearn

**Dataset**

Il dataset utilizzato proviene dal repository StatsBomb Open Data su GitHub. Può essere scaricato dal seguente link

--> StatsBomb Open Data: https://github.com/statsbomb/open-data

Per scaricare il dataset, esegui:

git clone https://github.com/statsbomb/open-data.git

Dopo il download, i file JSON contenenti i dati delle partite si trovano nella cartella open-data/data/events/.

**Struttura del Codice**

Il codice è suddiviso in più funzioni chiave:

_1. Caricamento ed Estrazione dei Dati_

load_events(match_ids, events_dir): Carica gli eventi delle partite selezionate dal dataset.

extract_shots(events): Estrae gli eventi di tipo "Shot" e organizza le informazioni chiave.

extract_goalkeepers(events): Estrae gli eventi relativi ai portieri e alle loro parate.

_2. Pre-processing e Feature Engineering_

preprocess_shots(df): Estrae coordinate, angoli e normalizza le feature dei tiri.

assign_goalkeeper(df, df_goalkeepers, match_selected): Associa a ciascun tiro il portiere avversario.

_3. Training dei Modelli per xG_

train_model(df, model, col_name, match_selected): Allena un modello (Logistic Regression, Random Forest o MLP) per predire il valore di xG di ciascun tiro.

print_model_metrics(model_name, metrics): Stampa le metriche di valutazione del modello.

plot_roc_curve(y_test, y_pred_proba, model_name): Genera il grafico della curva ROC.

_4. Data Visualization dei Tiri_

draw_pitch(ax): Disegna il campo da calcio.

visualize_shot_positions(df, match_selected): Mostra la distribuzione dei tiri.

visualize_final_shots_distribution(df, match_selected): Mostra la distribuzione dei tiri finali.

visualize_heatmap(df, match_selected): Genera una heatmap dei tiri in porta.

_5. Calcolo degli Expected Goals Saved (xGS)_

compute_goalkeeper_aggregates(df, df_goalkeepers, match_selected): Aggrega i dati per calcolare xGS.

get_goalkeeper_row_data(df, df_goalkeepers, match_selected): Organizza i dati in una tabella temporale.

_6. Data Visualization per i Portieri_

visualize_goalkeeper_stats(df, df_goalkeepers, match_selected): Confronta xG subiti e goal subiti.

visualize_goalkeeper_xgs(df, df_goalkeepers, match_selected): Mostra un'analisi comparativa degli xGS tra portieri.

visualize_goalkeeper_time_series(df, df_goalkeepers, match_selected): Mostra l'andamento temporale degli xG.

visualize_goalkeeper_xgs_over_time(df, df_goalkeepers, match_selected): Analizza l'evoluzione degli xGS nel tempo.

_7. Analisi Comparativa tra Partite_

comparative_goalkeeper_analysis(df_shots, df_goalkeepers, match1, match2): Confronta xG e goal subiti tra due partite.

comparative_goalkeeper_xgs_over_time(df_shots, df_goalkeepers, match1, match2): Confronta l'andamento degli xGS tra due match.

_8. Funzione Principale_

main(): Coordina il flusso di lavoro dal download del dataset all'analisi finale.

**Output e Visualizzazioni**

Dopo l'esecuzione dello script, verranno generati:

- Grafici per visualizzare le posizioni dei tiri
- Grafici di confronto tra xG concessi e gol subiti.
- Scatter plot per valutare il rendimento del portiere.
- Analisi temporale dell'andamento degli xGS.
- Grafici di confronto su più partite

**Contributi**

Se vuoi contribuire al miglioramento dello script, sentiti libero di fare un fork del repository e aprire una pull request.

**Autore**

Progetto sviluppato da Niccolò Beretta nell'ambito della tesi di laurea magistrale "Machine Learning e Data Analytics nel Contesto Calcistico" presso l'Università IULM nel corso Intelligenza Artificiale per l'Impresa e la Società.


