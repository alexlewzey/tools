from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from requests.exceptions import Timeout

from clmac.my_calendar import get_holiday, is_weekday


@patch("clmac.my_calendar.datetime")
def test_is_weekday(datetime_mock):
    sunday = datetime(2023, 12, 10)
    assert sunday.strftime("%A") == "Sunday"
    datetime_mock.now.return_value = sunday
    assert not is_weekday()
    thursday = datetime(2023, 12, 7)
    assert thursday.strftime("%A") == "Thursday"
    datetime_mock.now.return_value = thursday
    assert is_weekday()


@patch("clmac.my_calendar.requests")
def test_mock_response(requests_mock):
    response_mock = Mock(spec=["json", "status_code"])
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "mole": "lovely little boy",
        "ted": "lovely little boy",
    }
    requests_mock.get.side_effect = [Timeout, response_mock]
    with pytest.raises(Timeout):
        get_holiday()
    assert get_holiday()["mole"] == "lovely little boy"
    assert requests_mock.get.call_count == 2
