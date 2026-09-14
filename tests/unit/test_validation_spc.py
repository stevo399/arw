from datetime import date, datetime

from src.validation.spc import parse_spc_csv

TORNADO_CSV = (
    "Time,F_Scale,Location,County,State,Lat,Lon,Comments\n"
    "1945,UNK,3 NNE Lake Tanglewood,Randall,TX,35.10,-101.76,Brief touchdown. (AMA)\n"
    "0050,EF1,8 S Alberta,Stevens,MN,45.46,-96.06,Survey. (MPX)\n"
)
HAIL_CSV = "Time,Size,Location,County,State,Lat,Lon,Comments\n2010,275,2 N Moore,Cleveland,OK,35.38,-97.50,x\n"
WIND_CSV = "Time,Speed,Location,County,State,Lat,Lon,Comments\n2100,UNK,Town,County,OK,35.0,-97.0,x\n2105,70,Town,County,OK,35.1,-97.1,x\n"


def test_reports_before_12z_belong_to_the_next_utc_date():
    reports = parse_spc_csv(TORNADO_CSV, date(2026, 7, 12), "tornado")
    assert reports[0].time_utc == datetime(2026, 7, 12, 19, 45)
    assert reports[1].time_utc == datetime(2026, 7, 13, 0, 50)
    assert (reports[0].latitude, reports[0].longitude) == (35.10, -101.76)
    assert reports[0].magnitude is None and reports[1].magnitude == 1.0
    assert reports[1].state == "MN"


def test_hail_size_is_in_inches_and_wind_speed_in_knots():
    (hail,) = parse_spc_csv(HAIL_CSV, date(2026, 9, 12), "hail")
    assert hail.magnitude == 2.75
    unknown, known = parse_spc_csv(WIND_CSV, date(2026, 9, 12), "wind")
    assert unknown.magnitude is None and known.magnitude == 70.0


def test_header_only_file_has_no_reports():
    assert parse_spc_csv("Time,F_Scale,Location,County,State,Lat,Lon,Comments\n", date(2026, 4, 10), "tornado") == []
