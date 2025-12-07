
# Court Dynamics – Machine Learning Project

**Team Members:**  
- Yarkin Yavuz  
- Nisa Altay  
- Michelle Purevdagva

---

## 1. Introduction

This project analyzes **NBA player development and role dynamics** over multiple seasons using machine learning and data-driven exploratory analysis.

Instead of focusing only on a single prediction task, we adopt the **Court Dynamics** perspective:

- How do **player roles** emerge from statistical profiles?
- How do those roles and key performance metrics (e.g., True Shooting %) **evolve over time**?
- Which seasons are **anomalous** for a player relative to their own career?

Using a multi-season NBA dataset stored in `basketball.db`, we:

1. Engineer meaningful features such as **per-36-minute statistics**, **True Shooting (TS%)**, **age**, and **experience**.
2. Formulate a **regression task** to predict next-season TS% from current-season features.
3. Apply **KMeans clustering** to discover data-driven **player role archetypes**.
4. Explore **player trajectories** over time and highlight **unusually good or bad seasons** using simple anomaly detection.

The final outcome is a coherent, interpretable framework for discussing **how players play** (roles) and **how they develop** (trajectories and anomalies), grounded in quantitative analysis.

---

## 2. Methods

### 2.1. Data and schema

The project uses an SQLite database `basketball.db` containing multiple NBA seasons with at least the following tables:

- **`player_regular_season`**  
  Season-by-season box score statistics for each player (e.g., minutes, field goals, rebounds, assists, shooting statistics).

- **`players`**  
  Player-level biographical and static information (e.g., name, position, birthdate, first season).

We connect to the SQLite database using Python’s `sqlite3` library and load the main tables into pandas DataFrames.

### 2.2. Environment and tools

The analysis is implemented in a single Jupyter notebook: **`main.ipynb`**.

The environment is based on standard scientific Python tools:

- Python 3.x  
- `pandas`, `numpy`  
- `matplotlib`, `seaborn`  
- `scikit-learn`  
- `sqlite3`

An example conda-based setup:

```bash
conda create -n court-dynamics python=3.10 pandas numpy scikit-learn matplotlib seaborn
conda activate court-dynamics
jupyter notebook main.ipynb

#2.3. Feature engineering and preprocessing

We combine season-level and player-level information to build a player-season dataset.

Key engineered features include:
	•	Per-36-minute statistics
To normalize counting stats by playing time:
	•	pts_per_36 – points per 36 minutes
	•	reb_per_36 – rebounds per 36 minutes
	•	asts_per_36 – assists per 36 minutes
	•	stl_per_36 – steals per 36 minutes
	•	blk_per_36 – blocks per 36 minutes
	•	Shooting profile and efficiency
	•	three_rate: three-point attempt rate (3PA / FGA)
	•	ts: True Shooting %, computed in a numerically safe way to handle zero denominators
	•	Demographics and career stage
	•	birth_year, age in each season (season year – birth year)
	•	experience: years since the player’s first NBA season

Outlier handling

To stabilize models and clustering, we apply soft outlier clipping on a subset of continuous features
(e.g., per-36 stats, TS%, age, experience) using the 1st and 99th percentiles as caps.
This preserves most of the distribution while preventing extreme values from dominating.

Missing values

We use median imputation for missing values in the modeling datasets.
This approach is simple, robust, and avoids discarding rows with partially missing features.

Categorical encoding

The position variable is encoded via one-hot encoding (pd.get_dummies), generating indicators such as pos_G, pos_F, pos_C, etc.

#2.4. Regression task: predicting next-season TS%

We define a supervised regression problem:
	•	Input (X): Player-season features at season t
Per-36 stats, shooting profile (three_rate, current ts), age, experience, and positional dummies.
	•	Target (y): ts_next – True Shooting % in season t+1.

To construct this dataset:
	1.	Filter player-seasons to those above a minimum minutes threshold (e.g., 500 minutes) to reduce noise.
	2.	Create a “next-season” table shifted by one year and rename the TS column to ts_next.
	3.	Join current and next-season records on (player_id, year) so that each row contains:
	•	season t features and
	•	season t+1 TS% as the label (ts_next).

We then split this regression dataset into:
	•	Training+validation set and a held-out test set for final evaluation.
	•	Within training+validation, a further split into train and validation sets.

#2.5. Models and pipelines

We evaluate three regression models, each wrapped in a scikit-learn Pipeline with a StandardScaler:
	1.	Linear Regression
Ordinary least squares; serves as a simple baseline.
	2.	Ridge Regression
Linear regression with L2 regularization to control overfitting.
Main hyperparameter: regularization strength alpha.
	3.	Random Forest Regressor
Non-linear ensemble of decision trees.
Main hyperparameters: number of trees (n_estimators), maximum tree depth (max_depth), minimum samples per leaf (min_samples_leaf), etc.

All three models use the same feature set and are evaluated with the same train/validation/test splits.

#2.6. Role clustering with KMeans

To extract player role archetypes, we apply KMeans clustering on features that characterize on-court roles:
	•	pts_per_36 (scoring volume)
	•	reb_per_36
	•	asts_per_36
	•	stl_per_36
	•	blk_per_36
	•	three_rate (three-point attempt rate)
	•	ts (True Shooting %)

Steps:
	1.	Filter to player-seasons above the minutes threshold.
	2.	Impute missing values (medians).
	3.	Standardize the features.
	4.	Explore several values of k (e.g., 3–8) and compute silhouette scores as an internal clustering quality metric.
	5.	Choose a k that balances interpretability and silhouette score.
	6.	Fit KMeans with the chosen k and assign a cluster label to each player-season.

We then compute cluster-level mean feature values and inspect them to interpret each cluster as a role archetype.
A mapping from numeric cluster IDs to human-readable role names (e.g., “High-Usage Scoring Guard/Wing”, “Interior Finisher / Rim Protector”) is created and attached back to the data.

#2.7. Trajectories and anomaly detection

To incorporate temporal dynamics, we build a trajectories DataFrame where each row is a player-season with:
	•	Player identifier (ID, first name, last name)
	•	Season year
	•	Position
	•	Role cluster label (from KMeans)
	•	Key stats: TS%, pts_per_36, reb_per_36, asts_per_36, etc.

We sort by player and year to obtain chronological career trajectories.

For qualitative illustration, we:
	•	Select long-career players (e.g., players with ≥ 8 seasons in the dataset).
	•	Plot TS% and role cluster over time for at least one such player.

For anomaly detection, we compute a within-player TS% z-score:
	•	For each player, compute mean and standard deviation of TS% across their seasons.
	•	Define ts_z = (ts - mean_ts_player) / std_ts_player.
	•	Flag seasons with |ts_z| > 2 as anomalies:
	•	ts_z > 2: unusually efficient season relative to that player’s own career.
	•	ts_z < -2: unusually inefficient season.

We then list the most extreme positive and negative anomalies and, for at least one example, plot the player’s TS% trajectory over time.

⸻

#3. Experimental Design

Our experiments are structured around three core components: regression, role clustering, and trajectory/anomaly analysis.

#3.1. Experiment 1 – Next-season TS% regression

Goal

Assess how well current-season performance and context explain next-season TS% and compare different regression models.

Setup
	•	Inputs: per-36 stats, shooting profile (three_rate, ts), age, experience, positional dummies.
	•	Target: ts_next (True Shooting % in the subsequent season).
	•	Data splits:
	•	Train+validation vs. test.
	•	Train vs. validation within train+validation.

Models
	•	Linear Regression – baseline linear model.
	•	Ridge Regression – regularized linear model; alpha tuned via GridSearchCV with cross-validation.
	•	Random Forest Regressor – non-linear ensemble model; tuned with a small grid over n_estimators, max_depth, min_samples_leaf.

Metrics
	•	MAE (Mean Absolute Error)
	•	RMSE (Root Mean Squared Error)
	•	R² (Coefficient of Determination)

We compute these metrics on both validation and test sets.

#3.2. Experiment 2 – Player role clustering

Goal

Discover data-driven player role archetypes from season-level statistics, without pre-defined labels.

Setup
	•	Features: pts_per_36, reb_per_36, asts_per_36, stl_per_36, blk_per_36, three_rate, ts.
	•	Preprocessing: minutes filter, median imputation, feature scaling.

Procedure
	1.	Evaluate multiple values of k with the silhouette score.
	2.	Select a value of k that offers a good trade-off between cluster quality and interpretability.
	3.	Fit KMeans with this k and assign clusters to all eligible player-seasons.
	4.	Compute cluster-level mean feature values and the number of members per cluster.

Interpretation

Cluster means and example players in each cluster are inspected to interpret each cluster as an on-court role
(e.g., primary scorer, rim protector, 3-and-D wing, secondary playmaker, defensive role player).

#3.3. Experiment 3 – Trajectories and anomalies

Goal

Understand how player metrics and roles evolve across seasons and detect unusual seasons relative to a player’s own career baseline.

Setup
	•	Build a chronological trajectories table with per-season statistics and role clusters.
	•	Identify players with enough seasons (e.g., ≥ 8) for meaningful trajectories.

Procedure
	1.	For selected long-career players, plot:
	•	TS% vs. season year.
	•	Role cluster vs. season year (e.g., step plot).
	2.	For anomaly detection:
	•	Compute ts_z (within-player TS% z-score).
	•	Flag seasons with |ts_z| > 2 as anomalies.
	•	List the most extreme positive and negative anomalies.
	3.	For at least one extreme anomaly, visualize the full TS% trajectory to contextualize the anomalous season.

Evaluation

This experiment is primarily interpretive.
We judge success by:
	•	The clarity and interpretability of the patterns,
	•	The plausibility of identified anomalies,
	•	The usefulness of insights for understanding player development and role changes.

⸻

#4. Results

#4.1. Regression results

On the TS% regression task:
	•	All three models (Linear, Ridge, Random Forest) achieve reasonable predictive performance.
	•	Ridge Regression and Random Forest generally outperform plain Linear Regression in terms of MAE, RMSE, and R² on validation and test sets.
	•	Detailed numerical metrics are reported in the notebook’s final_results table.

Key observations
	•	Adding L2 regularization (Ridge) improves generalization compared to unregularized Linear Regression.
	•	Random Forest captures some non-linear patterns, but its advantage over Ridge is modest, suggesting that the relationship between our features and next-season TS% is relatively smooth.
	•	Overall, next-season TS% is moderately predictable from current-season features, which aligns with the intuition that shooting efficiency depends on both stable skills and changing context (team, role, injuries, etc.).

#4.2. Player role clusters

The KMeans clustering reveals several distinct role archetypes.
Based on average feature profiles and example players, we interpret clusters along lines such as:
	•	Interior Finisher / Rim Protector
High rebounding and blocks per 36, low three-point rate, moderate scoring.
Typically traditional bigs who protect the paint and finish near the rim.
	•	High-Usage Scoring Guard/Wing
Very high pts_per_36, elevated three-point rate, strong TS%.
Primary offensive options carrying a large scoring load.
	•	3-and-D Wing
Moderate scoring, high three-point rate, solid steals/blocks.
Players who space the floor and contribute on perimeter defense.
	•	Secondary Playmaker / Connector
Balanced scoring, relatively high assists per 36.
Often secondary ball-handlers who keep the offense flowing and create for others.
	•	Low-Usage Defensive Role Player
Low scoring and usage, but decent rebounding and defensive activity.
Complementary players whose value is mainly defensive and in non-box-score contributions.

(Exact cluster count and labels depend on the selected k and the data.)

Figures (suggested for the images/ folder)
	•	Cluster summary heatmap or table (e.g., images/role_cluster_summary.png).
	•	Scatter plot of TS% vs. points per 36, colored by cluster (e.g., images/pts_ts_scatter.png).

4.3. Trajectories and anomalies

From the trajectory and anomaly analysis:
	•	Long-career players show diverse patterns:
	•	Some have steady improvements in TS% over time.
	•	Others reach a strong efficiency peak and then decline.
	•	A few maintain relatively stable efficiency across many seasons.
	•	TS% z-score anomalies highlight:
	•	Breakout seasons, where a player’s TS% is much higher than their own typical level.
	•	Down years, where efficiency drops sharply relative to their career baseline.

Figures (suggested for the images/ folder)
	•	Example TS% trajectory for a long-career player (e.g., images/example_ts_trajectory.png), indicating where anomalous seasons occur.
	•	Tables showing top positive and negative TS anomalies, including player name, season year, TS%, TS z-score, and current role cluster.

These results support a nuanced view of player development:
roles and efficiencies are dynamic, and some seasons clearly stand out as turning points or outliers.

⸻

#5. Conclusions

#5.1. Main takeaways

This project demonstrates how a combination of classical machine learning techniques and careful feature engineering can yield a rich, interpretable picture of NBA player development and role dynamics:
	•	Regression models show that next-season TS% is moderately predictable from current-season per-36 stats, shooting profile, age, experience, and position. Regularized and non-linear models improve generalization compared to a simple linear baseline.
	•	KMeans clustering on per-36 and efficiency features naturally recovers intuitive player role archetypes such as primary scorers, rim protectors, 3-and-D wings, secondary playmakers, and defensive role players.
	•	Trajectory and anomaly analysis reveals that players follow diverse evolution patterns and that certain seasons are significantly better or worse than a player’s career norm, highlighting breakout years, role shifts, and declines.

Overall, the Court Dynamics framework connects how players play (roles), how they evolve (trajectories), and when they deviate from expectations (anomalies) in a unified, data-driven way.

#5.2. Limitations and future work

There are several limitations and natural extensions:
	•	Team context
We did not fully exploit team-level information (e.g., team offensive rating, pace, win–loss record). Incorporating such variables could help explain changes in roles and efficiency.
	•	Defensive metrics and advanced stats
We primarily focused on traditional box score and basic shooting efficiency. Including advanced stats (on/off metrics, impact ratings, lineup data) could refine role definitions and trajectory analysis.
	•	Richer trajectory modeling
Our trajectory analysis is mostly descriptive. More advanced sequence models (e.g., Hidden Markov Models, RNN-based approaches) could cluster full career paths and better capture phase transitions.
	•	Broader targets
We concentrated on TS% as the regression target. Future work could jointly model multiple outcomes (usage, assist rates, defensive impact) to capture a more complete view of player value.

Despite these limitations, the current pipeline already provides a clear, interpretable baseline for Court Dynamics and can serve as a foundation for more advanced analyses in future work.

#Repository Structure

A suggested repository layout is:

.
├── basketball.db          # SQLite database with NBA data
├── main.ipynb             # Main Jupyter notebook (complete analysis)
├── images/                # Figures used in the README
│   ├── pts_ts_scatter.png
│   ├── role_cluster_summary.png
│   └── example_ts_trajectory.png
└── README.md              # This file

All figures in the images/ folder are generated directly from main.ipynb.
