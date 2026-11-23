; Build with build_installer.py after portable verification and packaging.
#ifndef AppVersion
  #error AppVersion must be supplied by build_installer.py
#endif
#ifndef BundleDir
  #error BundleDir must contain the verified portable payload
#endif

[Setup]
AppId={{4D45D0C1-A8FA-4B22-909C-D341902BBD46}
AppName=SpectraFit
AppVersion={#AppVersion}
AppPublisher=HyperProbe / SpectraFit contributors
DefaultDirName={localappdata}\Programs\SpectraFit
DefaultGroupName=SpectraFit
AllowNoIcons=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0.19041
OutputDir={#OutputDir}
OutputBaseFilename=SpectraFit-{#AppVersion}-windows-x64-setup
LicenseFile={#BundleDir}\LICENSE
InfoBeforeFile={#SourcePath}\INSTALLER-NOTES.txt
UninstallDisplayIcon={app}\SpectraFit.exe
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes
CloseApplications=no
RestartApplications=no
ChangesAssociations=no
ChangesEnvironment=no

[Tasks]
Name: desktopicon; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SpectraFit"; Filename: "{app}\SpectraFit.exe"; WorkingDir: "{app}"
Name: "{group}\User guide"; Filename: "{app}\README.html"
Name: "{group}\Uninstall SpectraFit"; Filename: "{uninstallexe}"
Name: "{autodesktop}\SpectraFit"; Filename: "{app}\SpectraFit.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\SpectraFit.exe"; Description: "Open SpectraFit"; Flags: nowait postinstall skipifsilent
