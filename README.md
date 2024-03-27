# OBS Detect - Object Detection and Masking Filter

<div align="center">

[![GitHub](https://img.shields.io/github/license/occ-ai/obs-detect)](https://github.com/occ-ai/obs-detect/blob/main/LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/occ-ai/obs-detect/push.yaml)](https://github.com/occ-ai/obs-detect/actions/workflows/push.yaml)
[![Total downloads](https://img.shields.io/github/downloads/occ-ai/obs-detect/total)](https://github.com/occ-ai/obs-detect/releases)
![Flathub](https://img.shields.io/flathub/downloads/com.obsproject.Studio.Plugin.BackgroundRemoval?label=Flathub%20Installs)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/occ-ai/obs-detect)](https://github.com/occ-ai/obs-detect/releases)
[![Discord](https://img.shields.io/discord/1200229425141252116)](https://discord.gg/KbjGU2vvUz)

</div>

A plugin for [OBS Studio](https://obsproject.com/) that allows you to detect many types of objects in any source, track them and apply masking.

If you like this work, which is given to you completely free of charge, please consider supporting it by sponsoring us on GitHub:

- https://github.com/sponsors/royshil
- https://github.com/sponsors/umireon

## Usage

<div align="center">
<video src="https://github.com/occ-ai/obs-backgroundremoval/assets/1067855/5ba5aae2-7ea2-4c90-ad45-fba5ccde1a4e" width="320"></video>
</div>



### GitHub Actions Set Up

To use code signing on GitHub Actions, the certificate and associated information need to be set up as _repository secrets_ in the GitHub repository's settings.

* First, the locally stored developer certificate needs to be exported from the macOS keychain:
    * Using the Keychain app on macOS, export these your certificates (Application and Installer) public _and_ private keys into a single .p12 file **protected with a strong password**
    * Encode the .p12 file into its base64 representation by running `base64 <NAME_OF_YOUR_P12_FILE>`
* Next, the certificate data and the password used to export it need to be set up as repository secrets:
    * `MACOS_SIGNING_APPLICATION_IDENTITY`: Name of the "Developer ID Application" signing certificate
    * `MACOS_SIGNING_INSTALLER_IDENTITY`: Name of "Developer ID Installer" signing certificate
    * `MACOS_SIGNING_CERT`: The base64 encoded `.p12` file
    * `MACOS_SIGNING_CERT_PASSWORD`: Password used to generate the .p12 certificate
* To also enable notarization on GitHub Action runners, the following repository secrets are required:
    * `MACOS_NOTARIZATION_USERNAME`: Your Apple Developer account's _Apple ID_
    * `MACOS_NOTARIZATION_PASSWORD`: Your Apple Developer account's _generated app password_
