SELECT p.name_given, b.g AS games_played, b.r AS runs, b.h AS hits, b.hr AS home_runs
FROM batting b
JOIN player p ON b.player_id = p.player_id
WHERE b.g = (SELECT MAX(g) FROM batting)
   OR b.r = (SELECT MAX(r) FROM batting)
   OR b.h = (SELECT MAX(h) FROM batting)
   OR b.hr = (SELECT MAX(hr) FROM batting)
ORDER BY b.g DESC, b.r DESC, b.h DESC, b.hr DESC
