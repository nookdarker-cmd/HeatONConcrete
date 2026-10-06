from datetime import datetime

import ee
from flask import Flask, jsonify, render_template, request

PROJECT_ID = 'hicm-510809'
ee.Initialize(project=PROJECT_ID)

app = Flask(__name__)

THAILAND = ee.Geometry.Rectangle([97.3, 5.6, 105.7, 20.5])
MAX_DAYS = 366 * 5
_province_names = None


def parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d')


def prep(img):
    """คำนวณ LST (°C) + NDVI และตัดเมฆ/เงาเมฆ"""
    qa = img.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    optical = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    ndvi = optical.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    lst = (img.select('ST_B10').multiply(0.00341802).add(149.0)
           .subtract(273.15).rename('LST'))
    return (lst.addBands(ndvi).updateMask(mask)
            .copyProperties(img, ['system:time_start']))


def get_collection(start, end, region):
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    return (l8.merge(l9)
            .filterBounds(region)
            .filterDate(start, end)
            .map(prep))


def provinces_fc():
    return (ee.FeatureCollection('FAO/GAUL/2015/level1')
            .filter(ee.Filter.eq('ADM0_NAME', 'Thailand')))


def get_province_names():
    global _province_names
    if _province_names is None:
        _province_names = provinces_fc().aggregate_array('ADM1_NAME').sort().getInfo()
    return _province_names


def build_region(data):
    """คืน (พื้นที่, scale เป็นเมตร) ตามโหมดที่ผู้ใช้เลือก"""
    if data.get('province'):
        if data['province'] not in get_province_names():
            raise ValueError('ไม่พบจังหวัดนี้')
        geom = (provinces_fc()
                .filter(ee.Filter.eq('ADM1_NAME', data['province']))
                .geometry())
        return geom, 300
    g = data.get('geometry')
    if g:
        if g.get('type') not in ('Polygon', 'MultiPolygon'):
            raise ValueError('รูปทรงไม่ถูกต้อง')
        return ee.Geometry(g), 120
    lat = float(data['lat'])
    lon = float(data['lon'])
    return ee.Geometry.Point([lon, lat]).buffer(200), 30


def monthly_series(region, scale, start, end):
    """ค่าเฉลี่ยรายเดือนของ LST/NDVI ในพื้นที่ (ใช้ median ของภาพในเดือนนั้น)"""
    start_d = ee.Date(start)
    n_months = ee.Date(end).difference(start_d, 'month').ceil()
    col = get_collection(start, end, region)

    def per_month(m):
        s = start_d.advance(ee.Number(m), 'month')
        sub = col.filterDate(s, s.advance(1, 'month'))

        def make():
            vals = sub.median().reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=scale,
                bestEffort=True, maxPixels=1e9)
            return ee.Feature(None, {
                'date': s.format('YYYY-MM'),
                'LST': vals.get('LST'),
                'NDVI': vals.get('NDVI'),
            })

        return ee.Feature(ee.Algorithms.If(
            sub.size().gt(0), make(), ee.Feature(None, {})))

    months = ee.List.sequence(0, n_months.subtract(1))
    fc = (ee.FeatureCollection(months.map(per_month))
          .filter(ee.Filter.notNull(['LST'])))
    rows = [f['properties'] for f in fc.getInfo()['features']]
    rows.sort(key=lambda r: r['date'])
    return rows


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/tiles')
def tiles():
    try:
        start = request.args.get('start', '2025-01-01')
        end = request.args.get('end', '2025-12-31')
        parse_date(start)
        parse_date(end)

        col = get_collection(start, end, THAILAND)
        lst = col.select('LST').median()
        ndvi = col.select('NDVI').median()

        lst_id = lst.getMapId({'min': 25, 'max': 50,
                               'palette': ['blue', 'yellow', 'red']})
        ndvi_id = ndvi.getMapId({'min': 0, 'max': 0.8,
                                 'palette': ['brown', 'yellow', 'green']})
        return jsonify({
            'lst': lst_id['tile_fetcher'].url_format,
            'ndvi': ndvi_id['tile_fetcher'].url_format,
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/provinces')
def provinces():
    try:
        return jsonify(get_province_names())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/province_shape')
def province_shape():
    try:
        name = request.args.get('name', '')
        if name not in get_province_names():
            return jsonify({'error': 'ไม่พบจังหวัดนี้'}), 404
        geom = (provinces_fc()
                .filter(ee.Filter.eq('ADM1_NAME', name))
                .geometry().simplify(500))
        return jsonify(geom.getInfo())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/series', methods=['POST'])
def series():
    try:
        data = request.get_json(force=True)
        start = data.get('start', '2023-01-01')
        end = data.get('end', '2025-12-31')
        if (parse_date(end) - parse_date(start)).days > MAX_DAYS:
            return jsonify({'error': 'ช่วงเวลายาวเกิน 5 ปี'}), 400
        region, scale = build_region(data)
        return jsonify(monthly_series(region, scale, start, end))
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({'error': 'ข้อมูลที่ส่งมาไม่ถูกต้อง: ' + str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)