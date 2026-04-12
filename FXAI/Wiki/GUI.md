# GUI

FXAI GUI is the optional operator shell for users who want faster inspection, guided workflows, and a role-based view over the same artifact surfaces the terminal uses.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: the GUI makes live posture and artifact health readable quickly.
- Demo Trader: it turns runtime behavior into something easier to study.
- Backtester: it gives run builders and result browsing without memorizing every command.
- EA Researcher: it exposes plugin, report, and promotion surfaces with less friction.
- System Architect: it provides a coherent platform-control and recovery shell.

## What The GUI Is For

- first-class role workspaces that open the correct default operator surface for each user type
- overview dashboards and role dashboards with draggable, resizable panels
- runtime monitoring
- role-based workspaces
- Audit Lab, Backtest, and Offline Lab builders
- promotion and lineage inspection
- Research OS and recovery views

## How To Launch It

```bash
cd /path/to/FXAI/GUI
./start.sh
```

## How To Use It By Role

### Live Trader

Open `Live Overview`.

Use it for:
- health checks
- current runtime posture
- fast reading of why a pair is blocked or cautioned
- rearranging the dashboard so the panels you trust most stay in the top band for repeat reviews

### Demo Trader

Open `Demo Overview`.

Use it for:
- observing session changes
- comparing runtime posture against expected audit behavior
- saving a demo-specific dashboard layout that keeps scenarios and quick screens visible during study

### Backtester

Open `Backtest Builder`.

Use it for:
- quicker setup of focused test windows
- remembering exact commands without manual typing

### EA Researcher

Open `Research Workspace`.

Use it for:
- browsing plugin zoo and reports
- comparing promotion and lineage artifacts
- moving between research outputs and command recipes quickly
- dragging and resizing the workspace panels so commands, scenarios, and quick screens match your research flow

### System Architect

Open `Platform Control`.

Use it for:
- operator dashboard inspection
- research OS and branch health
- incident recovery guidance
- keeping incidents, commands, and quick screens in a layout tuned for governance work

## Example Case Scenarios

### Scenario: A live trader needs a one-minute trust check

What to do:
1. Open the Overview or Runtime Monitor.
2. Check health, freshness, and final posture.
3. Read the top reasons for the pair you care about.

### Scenario: A researcher wants to compare a new promotion candidate

What to do:
1. Open Research Workspace and Promotion Center.
2. Compare lineage, runtime artifacts, and report surfaces.
3. Use the generated command recipes to run the next validation step.

### Scenario: A system architect needs recovery guidance after missing artifacts

What to do:
1. Open Platform Control or Incident Center.
2. Identify which artifact or service is missing.
3. Run the recommended rebuild or recovery command from the GUI or terminal.

### Scenario: An operator wants a permanent custom dashboard

What to do:
1. Open `Overview`, `Live Overview`, `Demo Overview`, `Research Workspace`, or `Platform Control`.
2. Click `Customize`.
3. Drag panels by the handle chip, resize them with the corner handle or size controls, and let the GUI auto-save the layout.
4. Use `Reset Layout` if you need to return that dashboard to the shipped arrangement.

## When To Prefer The Terminal

- when you need exact repeatable command history
- when you are scripting or automating workflows
- when a subsystem needs low-level inspection beyond the surfaced views

## Next Pages

- [Getting Started](Getting%20Started.md)
- [Offline Lab](Offline%20Lab.md)
- [Runtime Control Plane](Runtime%20Control%20Plane.md)
