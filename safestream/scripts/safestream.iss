; Inno Setup 6 script for SafeStream Windows installer
[Setup]
AppName=SafeStream
AppVersion=1.0.0
AppPublisher=SafeStream
DefaultDirName={autopf}\SafeStream
DefaultGroupName=SafeStream
OutputDir=dist
OutputBaseFilename=SafeStream-1.0.0-Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\SafeStream\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SafeStream"; Filename: "{app}\SafeStream.exe"
Name: "{commondesktop}\SafeStream"; Filename: "{app}\SafeStream.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\SafeStream.exe"; Description: "Launch SafeStream now"; Flags: nowait postinstall skipifsilent

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
      'Please install Tesseract before running SafeStream:' + #13#10 +
      'https://github.com/UB-Mannheim/tesseract/wiki' + #13#10 + #13#10 +
      'After installing Tesseract, run this installer again.',
      mbInformation, MB_OK
    );
    Result := False;
  end else
    Result := True;
end;
