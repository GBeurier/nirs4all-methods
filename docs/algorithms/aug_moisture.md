# MoistureAugmenter

## Algorithm

Models water-band shifts due to changes in water activity ($a_w$) and
moisture content. Water exists in two states (free vs. hydrogen-bonded);
the relative fraction is governed by a sigmoid in $a_w$.

For each row:

1. $a_{w,i} = \mathrm{clip}(a_{w,\mathrm{ref}} + \Delta a_{w,i}, 0, 1)$
   where $\Delta a_{w,i}$ is uniform-sampled if a range is provided, else
   the fixed scalar.
2. $f_{\mathrm{free}} = \mathrm{free\_water\_fraction} \cdot \mathrm{sigmoid}(8(a_w - 0.5))$.
3. $f_{\mathrm{bound}} = 1 - f_{\mathrm{free}}$.
4. If `enable_shift`:
   - shift the water band centred at 1435 nm by $\mathrm{bound\_shift} \cdot f_{\mathrm{bound}}$.
   - shift the water band centred at 1930 nm by $0.8 \cdot \mathrm{bound\_shift} \cdot f_{\mathrm{bound}}$.
   Each shift is applied via `np.interp` with a Gaussian-weighted shift
   window centred at the band centre.
5. If `enable_intensity`:
   $$\mathrm{enhance} = \mathrm{moisture\_content} / 0.10 - 1$$
   $$X_{i,:} \mathrel{+}= \mathrm{enhance} \cdot 0.1 \cdot (g_{1435} + g_{1930}) \cdot |\overline{X_i}|$$
   where $g_c$ is a Gaussian region centred at wavelength $c$.

## Water-band peaks

The free and bound water peaks live at distinct wavelengths:

| Peak | Free water (nm) | Bound water (nm) |
|------|-----------------|------------------|
| First overtone | 1410 | 1460 |
| Combination    | 1920 | 1940 |

## Reference

`nirs4all.operators.augmentation.environmental.MoistureAugmenter`,
Büning-Pfaue (2003), Luck (1998).
