from app.repositories.inmemory import InMemoryJobRepository, InMemoryRunRepository

# Supabase-ready design:
# swap these with Supabase-backed implementations that satisfy same repository methods.
job_repo = InMemoryJobRepository()
run_repo = InMemoryRunRepository()
