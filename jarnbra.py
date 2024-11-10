# Nathan HK
# 2024-09-05

import os
from PIL import Image
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import sys
import time

"""
Command-line arguments:
1. Directory for images

PEP-8 compliant.
"""

skhn = (746, 861)  # Coordinates in browser for midpoint of image


def getCoordList(mappa):
    hnitlisti = []
    hnit_sv = {'h': [(64.167, 64.073, -21.649992, -22.041892, 40, 55),
                     (64.073, 64.033177, -21.871, -22.041892, 20, 20),
                     (64.200015, 64.167, -21.649992, -21.763, 10, 10)],
               'a': [(65.706074, 65.656070, -18.073935, -18.148693, 15, 15)],
               'r': [(64.030207, 63.954029, -22.467779, -22.592190, 15, 15)]}
    for svk in hnit_sv:
        for hnit_h in hnit_sv[svk]:
            diff = (hnit_h[0] - hnit_h[1], hnit_h[2] - hnit_h[3])
            for i in range(hnit_h[4]):
                lat = round(hnit_h[1] + (i * 2 + 1) * diff[0] /
                            (hnit_h[4] * 2), 6)
                for j in range(hnit_h[5]):
                    lon = round(hnit_h[3] + (j * 2 + 1) * diff[1] /
                                (hnit_h[5] * 2), 6)
                    hnitlisti.append((svk, lat, lon))

    for a in os.listdir(mappa):
        s = a.split('_')
        if len(s) != 3 or s[0] not in ['h', 'a', 'r'] or s[2][-4:] != '.png':
            continue
        tp = (s[0], float(s[1]), float(s[2][:-4]))
        if tp not in hnitlisti:
            hnitlisti.append(tp)

    return hnitlisti


def scrapeImages(mappa, hnitlisti):
    byrjun = time.time()
    if mappa[-1] != '/':
        mappa += '/'
    f = False
    for n in range(len(hnitlisti)):
        hnit = hnitlisti[n]
        try:
            z = open(mappa + hnit[0] + '_' + str(hnit[1]) + '_' +
                     str(hnit[2]) + '.png', 'rb')
            z.close()
        except FileNotFoundError:
            f = True
            break
    if f:
        driver = webdriver.Chrome()
        driver.set_window_size(1500, 1000)
        driver.get('https://ja.is/kort/?x=356954&y=408253&nz=17.00'
                   '&type=aerialnl')
        # Accept GDPR
        try:
            btn = driver.find_element(By.XPATH, '//a[@id="gdpr_banner_ok"]')
            btn.click()
        except NoSuchElementException:
            pass
        # Allow cookies
        try:
            btn = driver.find_element(By.XPATH, '//button[@class="ch2-btn '
                                      'ch2-allow-all-btn ch2-btn-primary"]')
            btn.click()
        except NoSuchElementException:
            pass
        leit = driver.find_element(By.XPATH, '//input[@id="mapq"]')
        for n in range(len(hnitlisti)):
            if n % 500 == 0:
                print(n, time.time() - byrjun)
            hnit = hnitlisti[n]
            try:
                # Does file exist?
                z = open(mappa + hnit[0] + '_' + str(hnit[1]) + '_' +
                         str(hnit[2]) + '.png', 'rb')
                z.close()
            except FileNotFoundError:
                # Input search term into search box
                leit.clear()
                leit.send_keys(str(hnit[1]) + ', ' + str(hnit[2] + 0.002))
                leit.send_keys(Keys.RETURN)
                time.sleep(2)  # Wait for images to load
                try:  # Place not found
                    driver.find_element(By.XPATH, '//div[@class="row '
                                        'not-found"]')
                except NoSuchElementException:
                    # Save and crop screenshot
                    driver.save_screenshot(mappa + hnit[0] + '_' +
                                           str(hnit[1]) + '_' + str(hnit[2]) +
                                           '.png')
                    skmynd = Image.open(mappa + hnit[0] + '_' + str(hnit[1]) +
                                        '_' + str(hnit[2]) + '.png')
                    skmynd = skmynd.crop((skhn[0] - 512, skhn[1] - 512,
                                          skhn[0] + 512, skhn[1] + 512))
                    skmynd.save(mappa + hnit[0] + '_' + str(hnit[1]) + '_' +
                                str(hnit[2]) + '.png')
                time.sleep(1)
        driver.close()
    print(time.time() - byrjun)


if __name__ == '__main__':
    hnitlisti = getCoordList(sys.argv[1])
    scrapeImages(sys.argv[1], hnitlisti)
