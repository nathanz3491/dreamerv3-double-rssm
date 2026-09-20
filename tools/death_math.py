"""Closed-form survival bounds for Craftax, derived from game_logic.py."""
# dexterity=1, strength=1 at spawn
MAXF = MAXD = MAXE = 7 + 2*1     # get_max_food/drink/energy
MAXH = 8 + 1                     # get_max_health
COEFF = 1.0 - 0.125*(1-1)        # intrinsic_decay_coeff at dexterity 1

# stat -1 when its counter exceeds the threshold, counter resets
food_period   = (25 + 1) / COEFF   # hunger  > 25
drink_period  = (20 + 1) / COEFF   # thirst  > 20
energy_period = (30 + 1) / COEFF   # fatigue > 30
health_period = (15 + 1)           # recover < -15 while any necessity == 0

print(f"max food/drink/energy = {MAXF}, max health = {MAXH}\n")
print(f"{'stat':<8}{'-1 every':>10}{'hits 0 at':>12}")
for name, per in (("food", food_period), ("drink", drink_period), ("energy", energy_period)):
    print(f"{name:<8}{per:>9.0f}s{MAXF*per:>11.0f}s")

first = min(MAXF*food_period, MAXF*drink_period, MAXF*energy_period)
print(f"\nfirst necessity fails at   {first:.0f} steps  (drink)")
print(f"health drains -1 every     {health_period:.0f} steps thereafter")
print(f"NEGLECT CEILING (death)    {first + MAXH*health_period:.0f} steps")
print(f"\nobserved agent death       ~278 steps")
print(f"=> agent reaches {278/(first + MAXH*health_period)*100:.0f}% of the do-nothing ceiling")
