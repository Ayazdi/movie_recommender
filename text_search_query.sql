SELECT 
  movies.title,
  ts_rank_cd(
    to_tsvector('english', movies.title),
    to_tsquery('''lord of the rings'':*')
  ) AS score
FROM movies
WHERE to_tsvector('english', movies.title) @@ to_tsquery('''lord of the rings'':*')
ORDER BY score DESC;
