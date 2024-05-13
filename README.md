# OBS Detect - Object Detection and Masking Filter

<div align="center">

[![GitHub](https://img.shields.io/github/license/occ-ai/obs-detect)](https://github.com/occ-ai/obs-detect/blob/main/LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/occ-ai/obs-detect/push.yaml)](https://github.com/occ-ai/obs-detect/actions/workflows/push.yaml)
[![Total downloads](https://img.shields.io/github/downloads/occ-ai/obs-detect/total)](https://github.com/occ-ai/obs-detect/releases)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/occ-ai/obs-detect)](https://github.com/occ-ai/obs-detect/releases)
[![Discord](https://img.shields.io/discord/1200229425141252116)](https://discord.gg/KbjGU2vvUz)

</div>

A plugin for [OBS Studio](https://obsproject.com/) that allows you to detect many types of objects in any source, track them and apply masking.

If you like this work, which is given to you completely free of charge, please consider supporting it by sponsoring us on GitHub:

- https://github.com/sponsors/royshil
- https://github.com/sponsors/umireon

This work uses the great contributions from [EdgeYOLO-ROS](https://github.com/fateshelled/EdgeYOLO-ROS) and [PINTO-Model-Zoo](https://github.com/PINTO0309/PINTO_model_zoo). The SORT algorithm is taken from https://github.com/yasenh/sort-cpp under the GPL license.

## Usage

<div align="center">
<a href="https://youtu.be/LrbUrvaGreQ"><img width="40%" src="https://github.com/occ-ai/obs-detect/assets/441170/b8e7367e-c1b0-4c7e-b0df-af45ead87199" /></a><br/>
  (2 Minute)
</div>

- Add the "Detect" filter to any source with an image (Media, Browser, VLC, Image, etc.)
- Enable "Masking" or "Tracking"

Use Detect to track your pet, or blur out people in your video!

More information and usage tutorials to follow soon.

## Features

Current features:

- Detect over 80 categories of objects, using an efficient model ([EdgeYOLO](https://github.com/LSH9832/edgeyolo))
- 3 Model sizes: Small, Medium and Large
- Control detection threshold
- Select object category filter (e.g. find only "Person")
- Masking: Blur, Solid color, Transparent, output binary mask (combine with other plugins!)
- Tracking: Single object / All objects, Zoom factor, smooth transition

Roadmap features:
- Precise object mask, beyond bounding box
- Implement SORT tracking for smoothness
- Multiple object category selection (e.g. Dog + Cat + Duck)
- Make available detection information for other plugins through settings
- More real-time models choices

## Building

The plugin was built and tested on Mac OSX  (Intel & Apple silicon), Windows and Linux.

Start by cloning this repo to a directory of your choice.

### Mac OSX

Using the CI pipeline scripts, locally you would just call the zsh script. By default this builds a universal binary for both Intel and Apple Silicon. To build for a specific architecture please see `.github/scripts/.build.zsh` for the `-arch` options.

```sh
$ ./.github/scripts/build-macos -c Release
```

#### Install
The above script should succeed and the plugin files (e.g. `obs-ocr.plugin`) will reside in the `./release/Release` folder off of the root. Copy the `.plugin` file to the OBS directory e.g. `~/Library/Application Support/obs-studio/plugins`.

To get `.pkg` installer file, run for example
```sh
$ ./.github/scripts/package-macos -c Release
```
(Note that maybe the outputs will be in the `Release` folder and not the `install` folder like `pakage-macos` expects, so you will need to rename the folder from `build_x86_64/Release` to `build_x86_64/install`)

### Linux (Ubuntu)

Use the CI scripts again
```sh
$ ./.github/scripts/build-linux.sh
```

Copy the results to the standard OBS folders on Ubuntu
```sh
$ sudo cp -R release/RelWithDebInfo/lib/* /usr/lib/x86_64-linux-gnu/
$ sudo cp -R release/RelWithDebInfo/share/* /usr/share/
```
Note: The official [OBS plugins guide](https://obsproject.com/kb/plugins-guide) recommends adding plugins to the `~/.config/obs-studio/plugins` folder.

### Windows

Use the CI scripts again, for example:

```powershell
> .github/scripts/Build-Windows.ps1 -Target x64
```

The build should exist in the `./release` folder off the root. You can manually install the files in the OBS directory.
