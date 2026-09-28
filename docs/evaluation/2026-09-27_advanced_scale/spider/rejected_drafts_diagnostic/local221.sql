SELECT t.team_long_name, COUNT(*) AS wins
FROM Match m
JOIN Team t ON m.home_team_api_id = t.team_api_id
WHERE m.home_team_goal > m.away_team_goal
GROUP BY t.team_long_name
ORDER BY wins DESC
LIMIT 10
