-- Capture log and dataset collection for the fabric inspection app.
-- Run once in Supabase > SQL Editor. Safe to re-run.

-- 1. Table schema
create table if not exists public.captures (
  id            uuid        primary key default gen_random_uuid(),
  captured_at   timestamptz not null,                -- browser clock at capture
  received_at   timestamptz not null default now(),  -- server clock at insert
  session_id    uuid        not null,                -- new per page load / Clear
  device_id     uuid        not null,                -- stable per browser (localStorage)
  verdict       text        not null check (verdict in ('defect', 'pass')),
  class_name    text        not null check (char_length(class_name) between 1 and 64),
  confidence    real        not null check (confidence between 0 and 1),
  probabilities jsonb       not null check (
    jsonb_typeof(probabilities) = 'object'
    and pg_column_size(probabilities) < 2048
  ),
  threshold     real        check (threshold between 0 and 1),
  model         text        check (char_length(model) <= 64),
  image_path    text        check (char_length(image_path) <= 256)
);

-- Ensure image_path exists if table was created earlier
alter table public.captures add column if not exists image_path text check (char_length(image_path) <= 256);

create index if not exists captures_session_idx on public.captures (session_id, captured_at);
create index if not exists captures_device_idx  on public.captures (device_id, captured_at);

-- Row Level Security
alter table public.captures enable row level security;

drop policy if exists "anon can insert captures" on public.captures;
create policy "anon can insert captures"
  on public.captures
  for insert
  to anon
  with check (true);

revoke all on public.captures from anon, authenticated;
grant insert on public.captures to anon;

-- 2. Storage Bucket for fabric capture images
insert into storage.buckets (id, name, public)
values ('fabric-captures', 'fabric-captures', true)
on conflict (id) do nothing;

-- Storage policies: allow anonymous visitors to upload captures
drop policy if exists "anon can upload captures" on storage.objects;
create policy "anon can upload captures"
  on storage.objects
  for insert
  to anon
  with check (bucket_id = 'fabric-captures');

-- Allow public viewing so images can be downloaded for dataset retraining
drop policy if exists "public can view captures" on storage.objects;
create policy "public can view captures"
  on storage.objects
  for select
  to public
  using (bucket_id = 'fabric-captures');
