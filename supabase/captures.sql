-- Capture log for the fabric inspection app.
-- Run once in Supabase > SQL Editor. Safe to re-run.
--
-- One row per capture from static/app.js. The browser uses the anon key, so the
-- anon role gets INSERT and nothing else: visitors can add rows but cannot read,
-- change or delete anyone's data. Read the table from the dashboard or with the
-- service_role key on a server, never from the browser.

create table if not exists public.captures (
  id            uuid        primary key default gen_random_uuid(),
  captured_at   timestamptz not null,                -- browser clock at capture
  received_at   timestamptz not null default now(),  -- server clock at insert
  session_id    uuid        not null,                -- new per page load / Clear
  device_id     uuid        not null,                -- stable per browser (localStorage)
  verdict       text        not null check (verdict in ('defect', 'pass')),
  class_name    text        not null check (char_length(class_name) between 1 and 64),
  confidence    real        not null check (confidence between 0 and 1),
  -- {"defect_free": 0.01, "hole": 0.93, ...}. Keyed by class name so a model
  -- with 6, 7 or 8 classes needs no schema change.
  probabilities jsonb       not null check (
    jsonb_typeof(probabilities) = 'object'
    and pg_column_size(probabilities) < 2048
  ),
  threshold     real        check (threshold between 0 and 1),
  model         text        check (char_length(model) <= 64)
);

create index if not exists captures_session_idx on public.captures (session_id, captured_at);
create index if not exists captures_device_idx  on public.captures (device_id, captured_at);

-- Row Level Security: with RLS on, every operation is denied unless a policy
-- allows it. Only INSERT gets a policy, so SELECT, UPDATE and DELETE stay
-- denied for anon and authenticated.
alter table public.captures enable row level security;

drop policy if exists "anon can insert captures" on public.captures;
create policy "anon can insert captures"
  on public.captures
  for insert
  to anon
  with check (true);  -- the column CHECKs above validate the row itself

-- Belt and braces: Supabase grants anon full table privileges by default, and
-- RLS already blocks them. Drop them too so a later policy mistake cannot
-- expose the table.
revoke all on public.captures from anon, authenticated;
grant insert on public.captures to anon;
