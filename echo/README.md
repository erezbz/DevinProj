# Pokémon Echo

A time-loop, skill-based game built on the open-source
[pokeemerald-expansion](https://github.com/rh-hideout/pokeemerald-expansion) engine.
Passion project: not for sale, and no ROM is distributed — `build.sh` produces it locally.

## Build
Requires the pokeemerald-expansion build dependencies (see its INSTALL.md) and an
arm-none-eabi toolchain (Arm GNU Toolchain 13.3 is known to work; pass
`TOOLCHAIN=/path/to/toolchain` if it is not on PATH).

```bash
./build.sh            # clones the engine next to this file, patches and builds
```

## What's in so far
- **Timing combat:** attacking opens a timing bar (Perfect 175% / Good 125% / Miss 75% damage);
  foe attacks open a guard bar (Perfect 25% / Good 60% damage taken).
- **Foe intents:** the action prompt shows what the foe plans this turn.
- **No grinding:** no EXP, no random encounters, Pokémon always obey; no Dynamax.
- **Three-day loop:** a bell tolls every 100 steps (Dawn/Day/Dusk/Night x 3 days, with
  matching lighting). After Day 3 Night, or on a blackout, the world resets; you keep your
  lead Pokémon, Echo Journal entries and Echo Shards. JOURNAL is in the Start menu.
- **New species (drawn by code, no AI):** Tickit (Bug/Steel), Chimell (Steel/Psychic), Gloamoth
  (Bug/Dark), Tidepup (Water), Bramblit (Grass/Fighting) and Solune, whose form, type and moves
  change each day of the loop (Fire -> Water -> Ghost). Regenerate with `tools/echo/gen_species.py`.
- **Island of Verran (new maps, authored from code):** Bellmoor town, Gloam Wood and Tideway Path
  (`tools/echo/gen_maps.py`, `tools/echo/gen_world.py`), plus three interiors.
- **Schedules + mystery:** NPCs and visible wild Pokémon appear by day/phase. Learn why Odette rings the
  Great Bell, find Corin's compass at the one Day 2 dawn low tide, and break the loop on the third night.
- **Echo Shards:** earned once (Kit, the dusk Gloamoth) and kept forever; each widens the timing windows.
- Settings live in `include/config/echo.h`. `ECHO_DEV_QUICKSTART` is a development shortcut.

## Testing
`runner.c` is a headless libmgba runner that plays the ROM from scripted inputs and saves
screenshots: `gcc -O2 -o runner runner.c -lmgba -lpng`, then
`./runner pokeemerald.gba out "w600 A w120 shot"`.
