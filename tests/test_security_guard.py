from main import is_out_of_scope_security_query


def test_detects_seed_key_recovery_request():
    assert is_out_of_scope_security_query("What is the seed/key recovery algorithm for Simos18?")


def test_detects_turkish_bypass_request():
    assert is_out_of_scope_security_query("ECU guvenligini bypass etmek icin RSA'yi nasil kirarim?")


def test_allows_benign_uds_question():
    assert not is_out_of_scope_security_query("What is UDS service 0x19 used for?")


def test_requires_both_topic_and_procedural_detail():
    assert not is_out_of_scope_security_query("What does a security access bypass mean in general?")
