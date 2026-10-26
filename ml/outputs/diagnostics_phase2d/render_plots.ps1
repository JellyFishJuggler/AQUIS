$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$root=(Resolve-Path -LiteralPath 'outputs/diagnostics_phase2d').Path
$dataDir=Join-Path $root 'visual_data'
$plotDir=Join-Path $root 'plots'
$sheetDir=Join-Path $root 'contact_sheets'
New-Item -ItemType Directory -Force -Path $plotDir,$sheetDir | Out-Null
$culture=[Globalization.CultureInfo]::InvariantCulture
$palette=@{
 raw=[Drawing.Color]::FromArgb(55,65,75); placeholder=[Drawing.Color]::FromArgb(142,68,173); spike=[Drawing.Color]::FromArgb(210,55,55); stuck=[Drawing.Color]::FromArgb(230,126,34); flip=[Drawing.Color]::FromArgb(200,45,140); burst=[Drawing.Color]::FromArgb(35,125,190)
}
function Draw-StationPlot([string]$Name,[string]$Class,[string]$CsvPath,[string]$OutPath){
 $rows=@(Import-Csv -LiteralPath $CsvPath)
 $all=@(); foreach($r in $rows){ if($r.gwl -ne ''){$all += [double]::Parse($r.gwl,$culture)} }
 if($all.Count -eq 0){return}
 $min=($all|Measure-Object -Minimum).Minimum; $max=($all|Measure-Object -Maximum).Maximum; if($max -le $min){$max=$min+1}; $pad=[Math]::Max(($max-$min)*0.05,0.05); $ymin=$min-$pad; $ymax=$max+$pad
 $W=1280;$H=760;$bmp=[Drawing.Bitmap]::new($W,$H);$gr=[Drawing.Graphics]::FromImage($bmp);$gr.SmoothingMode=[Drawing.Drawing2D.SmoothingMode]::AntiAlias;$gr.Clear([Drawing.Color]::White)
 $fontTitle=[Drawing.Font]::new('Arial',17,[Drawing.FontStyle]::Bold);$font=[Drawing.Font]::new('Arial',10);$fontSmall=[Drawing.Font]::new('Arial',9);$brush=[Drawing.Brushes]::Black
 $sample=Import-Csv -LiteralPath (Join-Path $root 'visual_sample_stations.csv')|Where-Object Station -eq $Name|Select-Object -First 1
 $counts=Import-Csv -LiteralPath (Join-Path $root 'visual_detector_counts.csv')|Where-Object Station -eq $Name|Select-Object -First 1
 $gr.DrawString($Name,$fontTitle,$brush,30,16)
 $sub="class: $Class | $($sample.selection) | placeholder=$($counts.placeholder_prefix_slots)  exact-spike=$($counts.exact_spike_mask_slots)  stuck=$($counts.stuck_window_slots)  flip=$($counts.flipflop_window_slots)  burst=$($counts.extreme_burst_points)"
 $gr.DrawString($sub,$fontSmall,[Drawing.Brushes]::DimGray,30,44)
 # Legend
 $legend=@(@('raw observed','raw'),@('placeholder prefix','placeholder'),@('exact-spike mask','spike'),@('stuck window','stuck'),@('flip-flop window','flip'),@('extreme-burst point','burst'))
 $lx=30;$ly=68
 foreach($item in $legend){$col=$palette[$item[1]];$sb=[Drawing.SolidBrush]::new($col);$gr.FillEllipse($sb,$lx,$ly+3,10,10);$gr.DrawString($item[0],$fontSmall,[Drawing.Brushes]::Black,$lx+14,$ly);$sb.Dispose();$lx+=([Math]::Max(112,$item[0].Length*7+24))}
 $xLeft=100;$xRight=1235;$yTop=105;$yBottom=680;$pw=$xRight-$xLeft;$ph=$yBottom-$yTop
 $gr.DrawRectangle([Drawing.Pens]::Black,$xLeft,$yTop,$pw,$ph)
 $obs=@();foreach($r in $rows){if($r.gwl_obs -ne '' -and [double]::Parse($r.gwl_obs,$culture) -eq 1.0 -and $r.gwl -ne ''){$obs+=,[pscustomobject]@{time=[DateTime]::Parse($r.time,$culture);value=[double]::Parse($r.gwl,$culture);row=$r}}}
 if($obs.Count -gt 0){$t0=$obs[0].time;$t1=$obs[-1].time;$span=[Math]::Max(($t1-$t0).TotalSeconds,1)}else{$t0=[DateTime]::MinValue;$t1=$t0.AddDays(1);$span=86400}
 for($k=0;$k -le 5;$k++){$f=$k/5.0;$y=$yBottom-[int]($f*$ph);$p1=[Drawing.Point]::new([int]$xLeft,[int]$y);$p2=[Drawing.Point]::new([int]$xRight,[int]$y);$gr.DrawLine([Drawing.Pens]::LightGray,$p1,$p2);$val=$ymin+$f*($ymax-$ymin);$gr.DrawString($val.ToString('0.00',$culture),$fontSmall,[Drawing.Brushes]::Black,5,$y-8)}
 for($k=0;$k -le 5;$k++){$f=$k/5.0;$x=$xLeft+[int]($f*$pw);$p1=[Drawing.Point]::new([int]$x,[int]$yTop);$p2=[Drawing.Point]::new([int]$x,[int]$yBottom);$gr.DrawLine([Drawing.Pens]::LightGray,$p1,$p2);$dt=$t0.AddSeconds($span*$f);$txt=$dt.ToString('yyyy-MM',$culture);$gr.DrawString($txt,$fontSmall,[Drawing.Brushes]::Black,$x-28,$yBottom+8)}
 $gr.DrawString('Date',$font,[Drawing.Brushes]::Black,[single]($xLeft+$pw/2-15),[single]($yBottom+34))
 $gr.TranslateTransform(24,[single]($yTop+$ph/2+65));$gr.RotateTransform(-90);$gr.DrawString('Groundwater level (m)',$font,[Drawing.Brushes]::Black,0,0);$gr.ResetTransform()
 $rawPen=[Drawing.Pen]::new($palette.raw,1.3);$prev=$null
 foreach($p in $obs){$x=$xLeft+[int](($p.time-$t0).TotalSeconds/$span*$pw);$y=$yBottom-[int](($p.value-$ymin)/($ymax-$ymin)*$ph);if($prev -and ($p.time-$prev.time).TotalHours -le 48){$px=$xLeft+[int](($prev.time-$t0).TotalSeconds/$span*$pw);$py=$yBottom-[int](($prev.value-$ymin)/($ymax-$ymin)*$ph);$p1=[Drawing.Point]::new([int]$px,[int]$py);$p2=[Drawing.Point]::new([int]$x,[int]$y);$gr.DrawLine($rawPen,$p1,$p2)};$prev=$p}
 $overlay=@(@('placeholder_prefix','placeholder',8),@('exact_spike_mask','spike',8),@('stuck_window','stuck',7),@('flipflop_window','flip',6),@('extreme_burst_point','burst',6))
 foreach($p in $obs){$x=$xLeft+[int](($p.time-$t0).TotalSeconds/$span*$pw);$y=$yBottom-[int](($p.value-$ymin)/($ymax-$ymin)*$ph);foreach($o in $overlay){if($p.row.($o[0]) -eq '1'){$bsh=[Drawing.SolidBrush]::new($palette[$o[1]]);$rr=[int]$o[2];$gr.FillEllipse($bsh,$x-$rr/2,$y-$rr/2,$rr,$rr);$bsh.Dispose()}}}
 $rawPen.Dispose();$fontTitle.Dispose();$font.Dispose();$fontSmall.Dispose();$gr.Dispose();$bmp.Save($OutPath,[Drawing.Imaging.ImageFormat]::Png);$bmp.Dispose()
}
$sample=Import-Csv -LiteralPath (Join-Path $root 'visual_sample_stations.csv')
$plotManifest=@()
foreach($r in $sample){$safe=[IO.Path]::GetFileNameWithoutExtension($r.selection);$data=Join-Path $dataDir ($r.Station -replace '[^A-Za-z0-9]+','_');$match=Get-ChildItem -LiteralPath $dataDir -Filter '*.csv'|Where-Object {$_.BaseName -like "*$(([BitConverter]::ToString([Security.Cryptography.MD5]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($r.Station))).Replace('-','').ToLower()).Substring(0,8))*"}|Select-Object -First 1;if(-not $match){throw "Missing plot data for $($r.Station)"};$png=Join-Path $plotDir ($match.BaseName+'_visual.png');Draw-StationPlot $r.Station $r.primary_classification $match.FullName $png}
$pngs=Get-ChildItem -LiteralPath $plotDir -Filter '*.png'|Sort-Object Name
foreach($r in $sample){$match=Get-ChildItem -LiteralPath $dataDir -Filter '*.csv'|Where-Object {$_.BaseName -like "*$(([BitConverter]::ToString([Security.Cryptography.MD5]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($r.Station))).Replace('-','').ToLower()).Substring(0,8))*"}|Select-Object -First 1;$counts=Import-Csv -LiteralPath (Join-Path $root 'visual_detector_counts.csv')|Where-Object Station -eq $r.Station|Select-Object -First 1;$png=Join-Path $plotDir ($match.BaseName+'_visual.png');$index=[array]::IndexOf(@($pngs.FullName),$png);$sheetN=[int]([Math]::Floor($index/6)+1);$plotManifest += [pscustomobject]@{Station=$r.Station;primary_classification=$r.primary_classification;selection=$r.selection;placeholder_prefix_slots=$counts.placeholder_prefix_slots;exact_spike_mask_slots=$counts.exact_spike_mask_slots;stuck_window_slots=$counts.stuck_window_slots;flipflop_window_slots=$counts.flipflop_window_slots;extreme_burst_points=$counts.extreme_burst_points;individual_plot=('plots/'+$match.BaseName+'_visual.png');contact_sheet=('contact_sheets/contact_sheet_{0:D2}.png' -f $sheetN)}}
for($i=0;$i -lt $pngs.Count;$i+=6){$sheet=[Drawing.Bitmap]::new(1920,1080);$g=[Drawing.Graphics]::FromImage($sheet);$g.Clear([Drawing.Color]::White);$end=[Math]::Min($i+5,$pngs.Count-1);for($j=$i;$j -le $end;$j++){$img=[Drawing.Image]::FromFile($pngs[$j].FullName);$k=$j-$i;$x=($k%3)*640;$y=[Math]::Floor($k/3)*540;$g.DrawImage($img,$x,$y,640,540);$img.Dispose()};$out=Join-Path $sheetDir ('contact_sheet_{0:D2}.png' -f [int]([Math]::Floor($i/6)+1));$sheet.Save($out,[Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$sheet.Dispose()}
$plotManifest|Export-Csv -LiteralPath (Join-Path $root 'visual_plot_manifest.csv') -NoTypeInformation -Encoding utf8
"Individual plots: $($pngs.Count); contact sheets: $((Get-ChildItem -LiteralPath $sheetDir -Filter '*.png').Count)"
