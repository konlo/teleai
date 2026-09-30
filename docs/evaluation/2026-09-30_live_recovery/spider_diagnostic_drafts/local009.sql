SELECT DISTINCT a1.city AS departure_city, a2.city AS arrival_city, a1.coordinates AS departure_coords, a2.coordinates AS arrival_coords
FROM flights f
JOIN airports_data a1 ON f.departure_airport = a1.airport_code
JOIN airports_data a2 ON f.arrival_airport = a2.airport_code
WHERE (a1.city->>'en' = 'Abakan' OR a1.city->>'ru' = 'Абакан' OR a2.city->>'en' = 'Abakan' OR a2.city->>'ru' = 'Абакан')
AND a1.coordinates IS NOT NULL AND a2.coordinates IS NOT NULL
