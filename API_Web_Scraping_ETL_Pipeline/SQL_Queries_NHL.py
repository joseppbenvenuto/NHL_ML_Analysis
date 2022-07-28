###########################################################################################################################################
# NEW CODE BLOCK - Drop all tables
###########################################################################################################################################

# DROP TABLES
teams_table_drop = "DROP TABLE IF EXISTS teams;"
time_table_drop = "DROP TABLE IF EXISTS time;"
season_stats_table_drop = "DROP TABLE IF EXISTS season_stats;"


###########################################################################################################################################
# NEW CODE BLOCK - Create all tables
###########################################################################################################################################

# CREATE TABLES
# DIMENSION TABLES
teams_table_create = ("""

    CREATE TABLE IF NOT EXISTS teams(
        team_id int NOT NULL PRIMARY KEY,
        team varchar NOT NULL
    );
    
""")


time_table_create = ("""

    CREATE TABLE IF NOT EXISTS time(
        season_year_range_id int NOT NULL PRIMARY KEY,
        season_year_range int NOT NULL
    );
    
""")

# FACT TABLE
season_stats_table_create = ("""

    CREATE TABLE IF NOT EXISTS season_stats(
        season_stats_id int NOT NULL PRIMARY KEY,
        team_id varchar NOT NULL,
        season_year_range_id varchar NOT NULL,
        games_played int NOT NULL,
        wins int NOT NULL,
        losses int NOT NULL,
        ot int NOT NULL,
        pts int NOT NULL,
        pt_pctg float NOT NULL,
        goals_per_game float NOT NULL,
        goals_against_per_game float NOT NULL,
        evgga_ratio float NOT NULL,
        power_play_percentage float NOT NULL,
        power_play_goals int NOT NULL,
        power_play_goals_against int NOT NULL,
        power_play_opportunities int NOT NULL,
        penalty_kill_percentage float NOT NULL,
        shots_per_game float NOT NULL,
        shots_allowed float NOT NULL,
        win_score_first float NOT NULL,
        win_opp_score_first float NOT NULL,
        win_lead_first_per float NOT NULL,
        win_lead_second_per float NOT NULL,
        win_outshoot_opp float NOT NULL,
        win_outshot_by_opp float NOT NULL,
        face_offs_taken int NOT NULL,
        face_offs_won int NOT NULL,
        face_offs_lost int NOT NULL,
        face_off_win_percentage float NOT NULL,
        shooting_pctg float NOT NULL,
        save_pctg float NOT NULL,
        failed_shots_per_game float NOT NULL,
        saves_per_game float NOT NULL,
        ties int NOT NULL,
        power_play_efficiency float NOT NULL,
        made_playoffs int NOT NULL,
        adjWins float NOT NULL
    );
    
""")


###########################################################################################################################################
# NEW CODE BLOCK - Insert records
###########################################################################################################################################

# INSERT RECORDS
teams_table_col_num = 2
teams_table_variables = '%s' + (',%s' * (teams_table_col_num - 1))
teams_table_insert = ("""

    INSERT INTO teams(
        team_id,
        team
    )
    VALUES (""" + teams_table_variables + """)
    ON CONFLICT (team_id)
        DO NOTHING;
        
""")

time_table_col_num = 2
time_table_variables = '%s' + (',%s' * (time_table_col_num - 1))
time_table_insert = ("""

    INSERT INTO time(
        season_year_range_id,
        season_year_range
    )
    VALUES (""" + time_table_variables + """)
    ON CONFLICT (season_year_range_id)
        DO NOTHING;
        
""")

season_stats_table_col_num = 37
season_stats_table_variables = '%s' + (',%s' * (season_stats_table_col_num - 1))
season_stats_table_insert = ("""

    INSERT INTO season_stats(
        season_stats_id,
        team_id,
        season_year_range_id,
        games_played,
        wins,
        losses,
        ot,
        pts,
        pt_pctg,
        goals_per_game,
        goals_against_per_game,
        evgga_ratio,
        power_play_percentage,
        power_play_goals,
        power_play_goals_against,
        power_play_opportunities,
        penalty_kill_percentage,
        shots_per_game,
        shots_allowed,
        win_score_first,
        win_opp_score_first,
        win_lead_first_per,
        win_lead_second_per,
        win_outshoot_opp,
        win_outshot_by_opp,
        face_offs_taken,
        face_offs_won,
        face_offs_lost,
        face_off_win_percentage,
        shooting_pctg,
        save_pctg,
        failed_shots_per_game,
        saves_per_game,
        ties,
        power_play_efficiency,
        made_playoffs,
        adjWins
    )
    VALUES (""" + season_stats_table_variables + """)
    ON CONFLICT (season_stats_id)
         DO NOTHING;
         
""")


###########################################################################################################################################
# NEW CODE BLOCK - Create View
###########################################################################################################################################

# Create sql nhl view
nhl_view_create = ("""

    CREATE VIEW nhl_view AS
    SELECT te.team,
        ti.season_year_range,
        se.games_played,
        se.wins,
        se.losses,
        se.ot,
        se.pts,
        se.pt_pctg,
        se.goals_per_game,
        se.goals_against_per_game,
        se.evgga_ratio,
        se.power_play_percentage,
        se.power_play_goals,
        se.power_play_goals_against,
        se.power_play_opportunities,
        se.penalty_kill_percentage,
        se.shots_per_game,
        se.shots_allowed,
        se.win_score_first,
        se.win_opp_score_first,
        se.win_lead_first_per,
        se.win_lead_second_per,
        se.win_outshoot_opp,
        se.win_outshot_by_opp,
        se.face_offs_taken,
        se.face_offs_won,
        se.face_offs_lost,
        se.face_off_win_percentage,
        se.shooting_pctg,
        se.save_pctg,
        se.failed_shots_per_game,
        se.saves_per_game,
        se.ties,
        se.power_play_efficiency,
        se.made_playoffs,
        se.adjWins
    FROM season_stats AS se LEFT JOIN teams AS te
    ON se.team_id = te.team_id
    LEFT JOIN time AS ti
    ON se.season_year_range_id = ti.season_year_range_id
    ORDER BY season_year_range ASC;

""")


###########################################################################################################################################
# NEW CODE BLOCK - Query lists
###########################################################################################################################################

# QUERY LISTS
create_table_queries = [
    teams_table_create, 
    time_table_create,
    season_stats_table_create
]

drop_table_queries = [
    teams_table_drop, 
    time_table_drop, 
    season_stats_table_drop
]
