SELECT p.name_given, b.g AS games, b.r AS runs, b.h AS hits, b.hr AS home_runs
FROM batting b
JOIN player p ON b.player_id = p.player_id
WHERE b.g IS NOT NULL AND b.r IS NOT NULL AND b.h IS NOT NULL AND b.hr IS NOT NULL
ORDER BY b.g DESC, b.r DESC, b.h DESC, b.hr DESC
LIMIT 10
