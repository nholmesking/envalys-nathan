# envalys-nathan
Hérna er forritasafn til að greina vegi á gervitunglamyndum. Ég (Nathan HK) bjó til það fyrir Envalys, en nú eftir uppsögnina mína er öllum leyft að nota það.

Markmið þessa verkefnis var að búa til þrívíddarmynd af vegunum. Ég gerði rúmlega helming áður en mér var sagt upp.

Listi yfir forrit (ég gaf þeim öll nöfn til þess að fólk rugli þeim ekki):
- **Járnbrá** sækir myndir frá vefnum og vistar þær.
- **Landbjartur** tekur myndir og vegagögn frá OpenStreetMap og breytir þeim í snið sem við getum notað til að þjálfa gervigreindarlíkan.
- **Unndís** skoðar myndir og þjálfar gervigreindarlíkan til að spá hvort svæði á myndinni séu með vegi eða ekki, og þá eyðir svæðunum án vega.
- **Kaðlín** tekur myndir og vegalista, og þjálfar gervigreindarlíkan til að spá staðsetningar vega.

## Járnbrá

Járnbrá býr til hnitlista og notar Selenium til að taka skjámyndir af [https://ja.is/kort/ Já.is], sem hún vistar á tölvunni.

Hún virkar ágætlega.

Ég veit ekki hvort þetta sé löglegt, en það er bara tímabundin lausn, og það er það besta sem ég gat gert án peninga.

## Landbjartur

Landbjartur tekur gervitunglamyndir frá Járnbrá og OpenStreetMap-gögn. Fyrir hverja mynd býr hann til 2x2 lista (4x4 í framtíðinni), og tölurnar í listanum eru 1 ef sambærilega svæðið á myndinni er með vegi, og 0 ef það er ekki. Hann vistar þessa lista á tölvunni til að geta notað aftur. Loksins breytir hann gögnunum í PyTorch-snið: myndsvæðin eru X-gögnin, og listarnir eru Y-gögnin.

Hann virkar ágætlega.

## Unndís

Unndís tekur gögnin frá Landbjarti og þjálfar gervigreindarlíkan til að spá hvort vegir séu til á myndunum. Þetta líkan er CNN sem spáir tölu milli 0 og 1 fyrir hvert myndsvæði. Eftir að þjálfunin er búin ætlar Unndís að eyða myndsvæðunum án vega, til að hjálpa Kaðlín.

Besta missitalan sem ég hef fengið er 0,34, og *precision* og *recall* eru bæði í kringum 80%. Því *recall* er langt mikilvægara fyrir Kaðlín þarf það að vera 99% eða meira.

## Kaðlín

Kaðlín tekur myndir (eftir skoðun Unndísar) og býr til vegform. Hún notar afbriðgi af Chamfers-fjarlægð.

Hún er alls ekki tilbúin.

## Orðabók

Hérna eru íslensk orð fyrir gervigreindarhugtök sem ég nota í þessu verkefni.
- **áttvigur** (no. kk.): *gradient*
- **endurbætir** (no. kk.): *optimizer*
- **greyping** (no. kvk.): *embedding*
- **lota** (no. kvk.): *epoch*
- **missitala** (no. kvk.): *loss*
- **námshraði** (no. kk.): *learning rate*
- **skriðþungi** (no. kk.): *momentum*
