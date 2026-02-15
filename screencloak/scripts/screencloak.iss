; Inno Setup 6 script for ScreenCloak Windows installer
[Setup]
AppName=ScreenCloak
AppVersion=1.0.0
AppPublisher=ScreenCloak
DefaultDirName={autopf}\ScreenCloak
DefaultGroupName=ScreenCloak
OutputDir=dist
OutputBaseFilename=ScreenCloak-1.0.0-Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\ScreenCloak\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ScreenCloak"; Filename: "{app}\ScreenCloak.exe"
Name: "{commondesktop}\ScreenCloak"; Filename: "{app}\ScreenCloak.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\ScreenCloak.exe"; Description: "Launch ScreenCloak now"; Flags: nowait postinstall skipifsilent

[Code]
// Check that Tesseract is installed before completing setup
function InitializeSetup(): Boolean;
var
  TesseractPath: String;
begin
  TesseractPath := 'C:\Program Files\Tesseract-OCR\tesseract.exe';
  if not FileExists(TesseractPath) then
  begin
    MsgBox(
      'Tesseract OCR is not installed.' + #13#10 + #13#10 +
      'Please install Tesseract before running ScreenCloak:' + #13#10 +
      'https://github.com/UB-Mannheim/tesseract/wiki' + #13#10 + #13#10 +
      'After installing Tesseract, run this installer again.',
      mbInformation, MB_OK
    );
    Result := False;
  end else
    Result := True;
end;
