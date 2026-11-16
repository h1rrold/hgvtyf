# SMPWeapons

Custom Paper plugin: 7 random weapons + the one-and-only **Heavy Hitter** mace, with 2D item models,
custom sounds and particle animations for every ability and passive.

Written against **Paper API 1.21.4** (Java 21). Newer Paper builds should load it; to compile against a newer API
just bump `paper-api` in `pom.xml`.

## Build

```
mvn package          # -> target/SMPWeapons.jar   (drop in /plugins)
```

Resource pack: upload `WeaponsPack.zip` somewhere public, then put the direct link (and `sha1sum` of the zip)
under `resource-pack:` in `plugins/SMPWeapons/config.yml`. Players get it on join. Without the pack everything
still works - weapons just look like netherite swords/axes and abilities fall back to vanilla sounds.

## Playing

* **Right-click** = Ability 1. **Sneak + right-click** = Ability 2 (unlocks at Tier II).
* Tier II: **3 player kills** or **killing a Wither**. Dying resets you to Tier I. Tier II = new model + glint and
  both cooldowns x0.75 (`tier2-cooldown-multiplier`).
* The SMP starts **off**. An admin runs `/start smp` to hand out weapons and enable abilities (`/end smp` to pause).
  Set `auto-start: true` in the config if you want it live immediately.
* Weapons are soulbound: can't be dropped, put in chests / ender chests / shulkers / bundles / hoppers / item frames.
  Anvil, enchanting table and grindstone are allowed. They survive death.

### Reroller
Craft (3x3): `Blaze Rod _ Blaze Rod / Ender Pearl, YOUR WEAPON, Ender Pearl / Breeze Rod _ Breeze Rod`.
It consumes your weapon and gives a soulbound Reroller; right-click it to get a different random weapon (tier kept).

### Commands
| Command | Who |
|---|---|
| `/trust <player>`, `/untrust <player>`, `/trust list` | everyone |
| `/start smp`, `/end smp` | admin (`weapons.admin`, default op) |
| `/give heavy hitter [player]`, `/remove heavy hitter` | admin (intercepted, vanilla `/give` is untouched) |
| `/reroll [player]`, `/upgrade [player]` | admin |
| `/weapon give <player> <weapon>` (alias `/wepon`), `remove`, `tier`, `list`, `info` | admin |
| `/heavyhitter give\|remove\|where` (alias `/hh`) | admin |

Trusted players are skipped by *your* abilities (one-way - they need to trust you back).

## Interpretations (change in code/config if you meant something else)
* **Heavy Hitter transfer**: killing the *current holder* moves the mace into the killer's hand. If the holder dies
  to a mob/environment they keep it. Duplicates are deleted ("ghosted") - only `data.yml`'s holder keeps one.
  Every other mace (vault loot, pick-ups, crafted) is deleted (`remove-other-maces`). It is *in addition to* the holder's weapon.
* **Tropical bubble**: each opponent inside is drained **once** (3 hearts) per bubble. Kiwi Fruit = renamed apple, 45/stack, 2 stacks max.
* **Ice Axe**: frozen 4 s (1 heart / 2 s), then 30 s chill (Slowness I + frost overlay) during which they can't be frozen again.
  Ability 2 (every 6th hit, double damage + Speed II for 2 s) is automatic. Powder-snow "walking" is done by nudging velocity.
* **Dynasty coins**: recharge = 10 s + 10 s per coin thrown (5 -> 60 s, 4 -> 50 s); one coin comes back 10 s after you run dry.
  "Hero of the Village x" is read as level X (10) - `hero-of-the-village-level`.
* **Inferno**: lava wall / fire zones are particles + damage zones (no real lava placed, so no griefing). Blaze Circle heals 5 hearts (`blaze-heal-hearts`).
* **Hellsfire**: ability hits are true damage + custom true fire for 12 s (3x vanilla's 4 s). The ground fire is a damage zone, not fire blocks.
* **Dragon's Fang**: only beneficial, non-healing effects are stretched to 10 min (speed, strength, fire res, night vision...).
* **Mini dragon**: a real 3D Ender Dragon model (69 cuboids, wings flap between two poses) exactly one block big (16x16x16 units). It is an item display, so it needs the resource pack. Use `/weapon dragon` to test it; if it flies backwards set `mini-dragon-yaw-offset: 180`. The coin is a flat sprite.
* Cooldowns for Inferno ability 2 weren't specified (45 s) - all cooldowns live in `config.yml`.

## Regenerating assets
`python3 tools/gen_textures.py` (weapons, 64x64), `python3 tools/gen_dragon.py` (3D dragon) and `python3 tools/gen_sounds.py` (needs Pillow, numpy, ffmpeg) rebuild every texture,
model JSON and `.ogg`. Swap any `.ogg` in `resourcepack/assets/weapons/sounds/` for your own audio - names stay the same.
