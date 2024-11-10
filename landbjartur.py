# Nathan HK
# 2024-09-05

import json
import math
import numpy as np
from PIL import Image
from pyrosm import OSM
from pyrosm import get_data
import sys
import time
import torch

import jarnbra

"""
Command-line arguments:
1. Directory for images
2. Program code

PEP-8 compliant.
"""

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def pix2coord(pix, hnit_br):
    x_t = hnit_br[1] + (pix[0] - 512) / 1024
    lon = math.degrees(x_t * 2 * math.pi / (2 ** 17) - math.pi)
    y_t = hnit_br[0] + (pix[1] - 512) / 1024
    lat = math.degrees(2 * (math.atan(math.exp(math.pi - y_t * 2 * math.pi /
                                               (2 ** 17))) - math.pi / 4))
    return (lat, lon)


def coord2pix(lat, lon, hnit_br):
    # Convert degrees to radians
    lon_radians = math.radians(lon)
    lat_radians = math.radians(lat)

    # Invert the calculation for x_t
    x_t = ((lon_radians + math.pi) * (2 ** 17)) / (2 * math.pi)
    # Calculate pix[0] (x-coordinate)
    pix_x = (x_t - hnit_br[1]) * 1024 + 512

    # Invert the calculation for y_t
    b = lat_radians / 2 + math.pi / 4
    a = math.tan(b)
    c = math.pi - math.log(a)
    y_t = c * (2 ** 17) / (2 * math.pi)
    # Calculate pix[1] (y-coordinate)
    pix_y = (y_t - hnit_br[0]) * 1024 + 512

    return (pix_x, pix_y)


def getRoadLists():
    fp = get_data('Iceland')
    veg_listi = {}
    osm_h = OSM(fp, bounding_box=[-22.140901, 63.847886,
                                  -21.152576, 64.390306])
    veg_listi['h'] = osm_h.get_network(network_type='driving')
    osm_a = OSM(fp, bounding_box=[-18.398071, 65.543087,
                                  -17.968359, 66.576398])
    veg_listi['a'] = osm_a.get_network(network_type='driving')
    osm_r = OSM(fp, bounding_box=[-22.735807, 63.883749,
                                  -22.359850, 64.090630])
    veg_listi['r'] = osm_r.get_network(network_type='driving')
    return veg_listi


def getRoadData_U(mappa, hnitlisti):
    byrjun = time.time()
    veg_listi = getRoadLists()
    try:
        jsonf = open(mappa + 'vegir.json', 'r')
        vegir_json = json.load(jsonf)
        jsonf.close()
    except FileNotFoundError:
        vegir_json = {}
    except json.JSONDecodeError:
        vegir_json = {}
    X_gogn = []
    y_gogn = []
    bd_all = {}
    for st in ['h', 'a', 'r']:
        vegir = veg_listi[st]
        bd = []
        for k in range(vegir.shape[0]):
            bns = vegir['geometry'][k].bounds
            bd.append(bns)
        bd_all[st] = bd
    for n in range(len(hnitlisti)):
        if n % 500 == 0:
            print(n, time.time() - byrjun)
        hnit = hnitlisti[n]

        # Open image
        try:
            gtm = Image.open(mappa + hnit[0] + '_' + str(hnit[1]) + '_' +
                             str(hnit[2]) + '.png')
            dilar = gtm.load()
            if tuple(dilar[0, 0]) == (4, 13, 23) and \
               tuple(dilar[1023, 1023]) == (4, 13, 23):
                print('Did not load', hnit)
        except FileNotFoundError:
            continue
        except OSError:
            print('OSError', hnit)
            continue

        # Convert coordinates
        if hnit[0] == 'd':
            vegir = veg_listi['h']
        else:
            vegir = veg_listi[hnit[0]]
        hnit_br = (1 / (2 * math.pi) * 2 ** 17 *
                   (math.pi - math.log(math.tan(math.pi / 4 +
                                                math.radians(hnit[1]) / 2))),
                   1 / (2 * math.pi) * 2 ** 17 *
                   (math.pi + math.radians(hnit[2])))

        # Roads
        try:
            ermedvegi = vegir_json[hnit[0] + '_' + str(hnit[1]) + '_' +
                                   str(hnit[2]) + '.png']
        except KeyError:
            print('KeyError', n)
            if hnit[0] == 'd':
                bd = bd_all['h']
            else:
                bd = bd_all[hnit[0]]
            ermedvegi = [0, 0, 0, 0]
            hNN = pix2coord(0, 0, hnit_br)
            hTT = pix2coord(1024, 1024, hnit_br)
            for v in range(vegir.shape[0]):
                bns = bd[v]
                if bns[0] > hnit[2] + 0.0014 or bns[1] > hnit[1] + 0.0008 or \
                   bns[2] < hnit[2] - 0.0014 or bns[3] < hnit[1] - 0.0008:
                    continue
                NW = coord2pix(bns[3], bns[0], hnit_br)
                SE = coord2pix(bns[1], bns[2], hnit_br)
                if NW[0] > 1024 or NW[1] > 1024 or SE[0] < 0 or SE[1] < 0:
                    continue
                for lina in vegir.loc[v, 'geometry'].geoms:
                    x, y = lina.xy
                    punktlisti = [(y[0], x[0]), (y[1], x[1])]
                    slope = (y[1] - y[0]) / (x[1] - x[0])
                    incpt = y[0] - slope * x[0]
                    if (hNN[1] < x[0] and hNN[1] > x[1]) or \
                       (hNN[1] > x[0] and hNN[1] < x[1]):
                        punktlisti.append((slope * hNN[1] + incpt, hNN[1]))
                    if (hnit[2] < x[0] and hnit[2] > x[1]) or \
                       (hnit[2] > x[0] and hnit[2] < x[1]):
                        punktlisti.append((slope * hnit[2] + incpt, hnit[2]))
                    if (hTT[1] < x[0] and hTT[1] > x[1]) or \
                       (hTT[1] > x[0] and hTT[1] < x[1]):
                        punktlisti.append((slope * hTT[1] + incpt, hTT[1]))
                    slope = (x[1] - x[0]) / (y[1] - y[0])
                    incpt = x[0] - slope * y[0]
                    if (hNN[0] < y[0] and hNN[0] > y[1]) or \
                       (hNN[0] > y[0] and hNN[0] < y[1]):
                        punktlisti.append((hNN[0], slope * hNN[0] + incpt))
                    if (hnit[1] < y[0] and hnit[1] > y[1]) or \
                       (hnit[1] > y[0] and hnit[1] < y[1]):
                        punktlisti.append((hnit[1], slope * hnit[1] + incpt))
                    if (hTT[0] < y[0] and hTT[0] > y[1]) or \
                       (hTT[0] > y[0] and hTT[0] < y[1]):
                        punktlisti.append((hTT[0], slope * hTT[0] + incpt))
                    for p in punktlisti:
                        if ermedvegi[0] == 1 and p[0] > hnit[1] and \
                           p[1] < hnit[2]:
                            continue
                        elif (ermedvegi[1] == 1 and p[0] > hnit[1] and
                              p[1] > hnit[2]):
                            continue
                        elif (ermedvegi[2] == 1 and p[0] < hnit[1] and
                              p[1] < hnit[2]):
                            continue
                        elif (ermedvegi[3] == 1 and p[0] < hnit[1] and
                              p[1] > hnit[2]):
                            continue
                        ct = coord2pix(p[0], p[1], hnit_br)
                        if ct[0] >= 0 and ct[0] <= 512 and ct[1] >= 0 and \
                           ct[1] <= 512:
                            ermedvegi[0] = 1
                        if ct[0] >= 512 and ct[0] <= 1024 and ct[1] >= 0 and \
                           ct[1] <= 512:
                            ermedvegi[1] = 1
                        if ct[0] >= 0 and ct[0] <= 512 and ct[1] >= 512 and \
                           ct[1] <= 1024:
                            ermedvegi[2] = 1
                        if ct[0] >= 512 and ct[0] <= 1024 and ct[1] >= 512 \
                           and ct[1] <= 1024:
                            ermedvegi[3] = 1
                        if sum(ermedvegi) == 4:
                            break
                    if sum(ermedvegi) == 4:
                        break
                if sum(ermedvegi) == 4:
                    break
            vegir_json[hnit[0] + '_' + str(hnit[1]) + '_' + str(hnit[2]) +
                       '.png'] = ermedvegi
        if hnit[0] != 'd':
            frummynd = np.array(gtm.getdata(band=2)).reshape(1024, 1024)
            X_gogn.append(torch.tensor(frummynd[:512, :512].reshape(1, 1,
                                                                    512, 512),
                                       dtype=torch.float16).to(device))
            X_gogn.append(torch.tensor(frummynd[:512, 512:].reshape(1, 1,
                                                                    512, 512),
                                       dtype=torch.float16).to(device))
            X_gogn.append(torch.tensor(frummynd[512:, :512].reshape(1, 1,
                                                                    512, 512),
                                       dtype=torch.float16).to(device))
            X_gogn.append(torch.tensor(frummynd[512:, 512:].reshape(1, 1,
                                                                    512, 512),
                                       dtype=torch.float16).to(device))
            y_gogn.append(torch.tensor(np.array(ermedvegi[0]).reshape(1, 1),
                                       dtype=torch.float16).to(device))
            y_gogn.append(torch.tensor(np.array(ermedvegi[1]).reshape(1, 1),
                                       dtype=torch.float16).to(device))
            y_gogn.append(torch.tensor(np.array(ermedvegi[2]).reshape(1, 1),
                                       dtype=torch.float16).to(device))
            y_gogn.append(torch.tensor(np.array(ermedvegi[3]).reshape(1, 1),
                                       dtype=torch.float16).to(device))
        gtm.close()
    outfile = open(mappa + 'vegir.json', 'w')
    json.dump(vegir_json, outfile)
    outfile.close()
    print(time.time() - byrjun)
    return X_gogn, y_gogn


def getRoadData_K(mappa, hnitlisti):
    byrjun = time.time()
    X_gogn = []
    y_gogn = []
    bd_all = {}
    for st in ['h', 'a', 'r']:
        vegir = veg_listi[st]
        bd = []
        for k in range(vegir.shape[0]):
            bns = vegir['geometry'][k].bounds
            bd.append(bns)
        bd_all[st] = bd
    for n in range(len(hnitlisti)):
        if n % 100 == 0:
            print(n, len(X_gogn), time.time() - byrjun)
        hnit = hnitlisti[n]
        
        # Open image
        try:
            gtm = Image.open(mappa + hnit[0] + '_' + str(hnit[1]) + '_' + str(hnit[2]) + '.png')
            dilar = gtm.load()
        except FileNotFoundError:
            continue
        except OSError:
            print('OSError', hnit)
            continue
            
        # Convert coordinates
        y_n = 1 / (2 * math.pi) * 2 ** 17 * (math.pi - math.log(math.tan(math.pi / 4 + math.radians(hnit[1]) / 2))) - 0.5
        y_s = 1 / (2 * math.pi) * 2 ** 17 * (math.pi - math.log(math.tan(math.pi / 4 + math.radians(hnit[1]) / 2))) + 0.5
        if hnit[0] == 'd':
            vegir = veg_listi['h']
        else:
            vegir = veg_listi[hnit[0]]
        hnit_br = (1 / (2 * math.pi) * 2 ** 17 * (math.pi - math.log(math.tan(math.pi / 4 + math.radians(hnit[1]) / 2))),
                   1 / (2 * math.pi) * 2 ** 17 * (math.pi + math.radians(hnit[2])))
        
        # Roads
        y_mynd = np.zeros((1, 1, 1024, 1024))
        if hnit[0] == 'd':
            bd = bd_all['h']
        else:
            bd = bd_all[hnit[0]]
        total_NW = pix2coord((0, 0), hnit_br)
        total_SE = pix2coord((1024, 1024), hnit_br)
        linulisti = []
        for v in range(vegir.shape[0]):
            bns = bd[v]
            if bns[0] > hnit[2] + 0.0014 or bns[1] > hnit[1] + 0.0008 or bns[2] < hnit[2] - 0.0014 or bns[3] < hnit[1] - 0.0008:
                continue
            NW = coord2pix(bns[3], bns[0], hnit_br)
            SE = coord2pix(bns[1], bns[2], hnit_br)
            if NW[0] > 1024 or NW[1] > 1024 or SE[0] < 0 or SE[1] < 0:
                continue
            for p in vegir.loc[v, 'geometry'].geoms:
                x, y = p.xy
                if y[0] > total_NW[0] and y[1] > total_NW[0]:
                    continue
                if y[0] < total_SE[0] and y[1] < total_SE[0]:
                    continue
                if x[0] < total_NW[1] and x[1] < total_NW[1]:
                    continue
                if x[0] > total_SE[1] and x[1] > total_SE[1]:
                    continue
                ct1 = coord2pix(y[0], x[0], hnit_br)
                ct2 = coord2pix(y[1], x[1], hnit_br)
                linulisti.append(list(ct1) + list(ct2))
        if hnit[0] != 'd' and len(mot) > 0:
            X_gogn.append(torch.tensor(np.transpose(np.array(gtm.getdata()).reshape(1024, 1024, 3)[:, :, 2:3], 
                                                    (2, 0, 1)).reshape(1, 1, 1024, 1024), dtype=torch.float32).to(device))
            y_gogn.append(torch.tensor(np.array(linulisti).reshape(1, -1, 2), dtype=torch.float32).to(device))
        gtm.close()
    print(time.time() - byrjun)
    return X_gogn, y_gogn


def getRoadData(forrit, mappa, hnitlisti):
    if mappa[-1] != '/':
        mappa += '/'
    if forrit == 'U':
        return getRoadData_U(mappa, hnitlisti)
    elif forrit == 'K':
        return getRoadData_K(mappa, hnitlisti)
    else:
        raise ValueError('Invalid program code')


if __name__ == '__main__':
    hnitlisti = jarnbra.getCoordList(sys.argv[1])
    getRoadData(sys.argv[2], sys.argv[1], hnitlisti)
