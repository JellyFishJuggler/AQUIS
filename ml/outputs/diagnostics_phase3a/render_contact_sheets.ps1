$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$root=(Resolve-Path -LiteralPath 'outputs/diagnostics_phase3a').Path
$data=Import-Csv -LiteralPath (Join-Path $root 'visual_data/series.csv')
$sample=Import-Csv -LiteralPath (Join-Path $root 'visual_sample_stations.csv')
$culture=[Globalization.CultureInfo]::InvariantCulture
$colors=@{raw=[Drawing.Color]::FromArgb(55,65,75);m1=[Drawing.Color]::FromArgb(142,68,173);m2=[Drawing.Color]::FromArgb(210,55,55);m3=[Drawing.Color]::FromArgb(35,125,190);stuck=[Drawing.Color]::FromArgb(230,126,34);flipflop=[Drawing.Color]::FromArgb(200,45,140);extreme_burst=[Drawing.Color]::FromArgb(35,160,100)}
function Draw-Panel($g,$rows,[string]$valueCol,[int]$x0,[int]$y0,[int]$w,[int]$h,[double]$ymin,[double]$ymax,[datetime]$t0,[datetime]$t1,[string]$title){
 $left=$x0+65;$right=$x0+$w-20;$top=$y0+28;$bottom=$y0+$h-48;$pw=$right-$left;$ph=$bottom-$top
 $g.DrawString($title,[Drawing.Font]::new('Arial',12,[Drawing.FontStyle]::Bold),[Drawing.Brushes]::Black,$x0+8,$y0+4)
 $g.DrawRectangle([Drawing.Pens]::Black,$left,$top,$pw,$ph)
 for($k=0;$k -le 4;$k++){$f=$k/4.0;$yy=$bottom-[int]($f*$ph);$g.DrawLine([Drawing.Pens]::LightGray,$left,$yy,$right,$yy);$val=$ymin+$f*($ymax-$ymin);$g.DrawString($val.ToString('0.0',$culture),[Drawing.Font]::new('Arial',8),[Drawing.Brushes]::Black,3,$yy-8)}
 for($k=0;$k -le 4;$k++){$f=$k/4.0;$xx=$left+[int]($f*$pw);$g.DrawLine([Drawing.Pens]::LightGray,$xx,$top,$xx,$bottom);$dt=$t0.AddTicks([long](($t1.Ticks-$t0.Ticks)*$f));$g.DrawString($dt.ToString('yyyy-MM'),[Drawing.Font]::new('Arial',8),[Drawing.Brushes]::Black,$xx-25,$bottom+5)}
 $prev=$null;$pen=[Drawing.Pen]::new($colors.raw,1.2)
 foreach($r in $rows){$v=$r.$valueCol;if($v -eq ''){$prev=$null;continue};$tm=[datetime]::Parse($r.time,$culture);$val=[double]::Parse($v,$culture);$xx=$left+[int](($tm-$t0).TotalSeconds/[math]::Max(1,($t1-$t0).TotalSeconds)*$pw);$yy=$bottom-[int](($val-$ymin)/($ymax-$ymin)*$ph);if($prev -and ($tm-$prev.tm).TotalHours -le 48){$g.DrawLine($pen,$prev.x,$prev.y,$xx,$yy)};$prev=[pscustomobject]@{tm=$tm;x=$xx;y=$yy}}
 # detector overlay points are positioned on raw value, including on after panel for comparison
 foreach($r in $rows){if($r.gwl -eq ''){continue};$tm=[datetime]::Parse($r.time,$culture);$val=[double]::Parse($r.gwl,$culture);$xx=$left+[int](($tm-$t0).TotalSeconds/[math]::Max(1,($t1-$t0).TotalSeconds)*$pw);$yy=$bottom-[int](($val-$ymin)/($ymax-$ymin)*$ph);foreach($key in @('m1','m2','m3','stuck','flipflop','extreme_burst')){if($r.$key -eq 'True' -or $r.$key -eq '1'){$b=[Drawing.SolidBrush]::new($colors[$key]);$g.FillEllipse($b,$xx-3,$yy-3,6,6);$b.Dispose()}}}
 $pen.Dispose();$g.DrawString('Groundwater level (m)',[Drawing.Font]::new('Arial',8),[Drawing.Brushes]::Black,$x0+5,$y0+$h-20)
}
$sheetDir=Join-Path $root 'contact_sheets';New-Item -ItemType Directory -Force -Path $sheetDir | Out-Null
$allnames=@($sample.Station)
for($start=0;$start -lt $allnames.Count;$start+=6){
 $bmp=[Drawing.Bitmap]::new(3000,2000);$g=[Drawing.Graphics]::FromImage($bmp);$g.SmoothingMode=[Drawing.Drawing2D.SmoothingMode]::AntiAlias;$g.Clear([Drawing.Color]::White)
 for($q=$start;$q -lt [math]::Min($start+6,$allnames.Count);$q++){
  $name=$allnames[$q];$s=$sample|Where-Object Station -eq $name|Select-Object -First 1;$rows=@($data|Where-Object Station -eq $name);if(!$rows.Count){continue}
  $times=@($rows|ForEach-Object {[datetime]::Parse($_.time,$culture)});$t0=($times|Measure-Object -Minimum).Minimum;$t1=($times|Measure-Object -Maximum).Maximum
  $vals=@($rows|Where-Object {$_.gwl -ne ''}|ForEach-Object {[double]::Parse($_.gwl,$culture)});if(!$vals.Count){continue};$lo=($vals|Measure-Object -Minimum).Minimum;$hi=($vals|Measure-Object -Maximum).Maximum;if($hi -le $lo){$hi=$lo+1};$pad=[math]::Max(($hi-$lo)*.04,.05);$lo-=$pad;$hi+=$pad
  $tile=$q-$start;$x=($tile%3)*1000;$y=[math]::Floor($tile/3)*1000
  $cntM1=(@($rows|Where-Object {$_.m1 -eq 'True' -or $_.m1 -eq '1'})).Count;$cntM2=(@($rows|Where-Object {$_.m2 -eq 'True' -or $_.m2 -eq '1'})).Count;$cntM3=(@($rows|Where-Object {$_.m3 -eq 'True' -or $_.m3 -eq '1'})).Count;$cntS=(@($rows|Where-Object {$_.stuck -eq 'True' -or $_.stuck -eq '1'})).Count;$cntF=(@($rows|Where-Object {$_.flipflop -eq 'True' -or $_.flipflop -eq '1'})).Count;$cntE=(@($rows|Where-Object {$_.extreme_burst -eq 'True' -or $_.extreme_burst -eq '1'})).Count
  $g.DrawString($name,[Drawing.Font]::new('Arial',14,[Drawing.FontStyle]::Bold),[Drawing.Brushes]::Black,$x+8,$y+5)
  $caption="class=$($s.primary_classification) | M1=$cntM1 M2=$cntM2 M3=$cntM3 | stuck=$cntS flip=$cntF burst=$cntE"
  $g.DrawString($caption,[Drawing.Font]::new('Arial',9),[Drawing.Brushes]::DimGray,$x+8,$y+27)
  Draw-Panel $g $rows 'gwl' $x ($y+45) 1000 440 $lo $hi $t0 $t1 'Before (raw observed)'
  Draw-Panel $g $rows 'gwl_after' $x ($y+490) 1000 440 $lo $hi $t0 $t1 'After (M1/M2 masked; M3 break, flags shown)'
 }
 $legend='Legend: raw=dark gray | M1 placeholder/early era=purple | M2 spike=red | M3 break=blue | stuck=orange | flip-flop=magenta | extreme burst=green'
 $g.DrawString($legend,[Drawing.Font]::new('Arial',11,[Drawing.FontStyle]::Bold),[Drawing.Brushes]::Black,20,1970)
 $out=Join-Path $sheetDir ('phase3a_contact_sheet_{0:D2}.png' -f [int]([math]::Floor($start/6)+1));$bmp.Save($out,[Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$bmp.Dispose()
}
"Contact sheets: $((Get-ChildItem -LiteralPath $sheetDir -Filter 'phase3a_contact_sheet_*.png').Count)"
