-- Run once in a fresh Supabase project. No service-role key in the app.
create table public.profiles (
 id uuid primary key references auth.users(id) on delete cascade,
 display_name text not null default '' check (char_length(display_name)<=80),
 preferences jsonb not null default '{}',
 bookmarks jsonb not null default '[]'
);
create table public.experiments (
 id uuid primary key default gen_random_uuid(),
 user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
 name text not null check(char_length(name) between 1 and 100),
 notes text not null default '' check(char_length(notes)<=4000),
 settings jsonb not null default '{}',
 metrics jsonb not null default '{}',
 object_path text,
 created_at timestamptz not null default now(),
 check (object_path is null or object_path = user_id::text || '/' || id::text || '.npz')
);
create index experiments_owner_created on public.experiments(user_id,created_at desc);
alter table public.profiles enable row level security;
alter table public.experiments enable row level security;
revoke all on public.profiles,public.experiments from anon;
grant select,update on public.profiles to authenticated;
grant select,insert,update,delete on public.experiments to authenticated;
create policy "Own profile read" on public.profiles for select to authenticated using (id=auth.uid());
create policy "Own profile update" on public.profiles for update to authenticated using (id=auth.uid()) with check(id=auth.uid());
create policy "Own experiments read" on public.experiments for select to authenticated using(user_id=auth.uid());
create policy "Own experiments insert" on public.experiments for insert to authenticated with check(user_id=auth.uid());
create policy "Own experiments update" on public.experiments for update to authenticated using(user_id=auth.uid()) with check(user_id=auth.uid());
create policy "Own experiments delete" on public.experiments for delete to authenticated using(user_id=auth.uid());
create function public.new_lab_user() returns trigger language plpgsql security definer set search_path='' as $$
begin
 insert into public.profiles(id,display_name) values(new.id,left(coalesce(new.raw_user_meta_data->>'display_name',''),80));
 return new;
end $$;
revoke all on function public.new_lab_user() from public;
create trigger lab_user_created after insert on auth.users for each row execute procedure public.new_lab_user();
-- Backfill users created before this migration.
insert into public.profiles(id,display_name)
 select id,left(coalesce(raw_user_meta_data->>'display_name',''),80) from auth.users on conflict do nothing;
insert into storage.buckets(id,name,public,file_size_limit,allowed_mime_types)
 values('experiments','experiments',false,12582912,array['application/octet-stream'])
 on conflict(id) do nothing;
create policy "Private results read" on storage.objects for select to authenticated
 using(bucket_id='experiments' and (storage.foldername(name))[1]=auth.uid()::text);
create policy "Private results insert" on storage.objects for insert to authenticated
 with check(bucket_id='experiments' and (storage.foldername(name))[1]=auth.uid()::text);
create policy "Private results delete" on storage.objects for delete to authenticated
 using(bucket_id='experiments' and (storage.foldername(name))[1]=auth.uid()::text);
