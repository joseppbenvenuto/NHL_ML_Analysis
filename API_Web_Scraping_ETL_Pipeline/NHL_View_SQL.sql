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

SELECT COUNT(*)
FROM nhl_view;

SELECT * 
FROM nhl_view;
