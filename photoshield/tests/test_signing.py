"""Owner signatures (design §5.4): valid, tampered, wrong purpose."""
from photoshield.provenance import signing as S


def test_valid_signature_verifies():
    sk, pk = S.new_key(b"\x01" * 32)
    msg = S.register_message("ab" * 32)
    assert S.verify(pk, S.sign(sk, msg), msg)


def test_flipped_message_or_signature_is_rejected():
    sk, pk = S.new_key(b"\x02" * 32)
    msg = S.register_message("cd" * 32)
    sig = S.sign(sk, msg)
    assert not S.verify(pk, sig, S.register_message("ce" + "cd" * 31))
    bad = ("0" if sig[0] != "0" else "1") + sig[1:]
    assert not S.verify(pk, bad, msg)


def test_other_key_is_rejected():
    sk, _ = S.new_key(b"\x03" * 32)
    _, other_pk = S.new_key(b"\x04" * 32)
    msg = S.register_message("ef" * 32)
    assert not S.verify(other_pk, S.sign(sk, msg), msg)


def test_purpose_prefix_prevents_cross_use():
    sk, pk = S.new_key(b"\x05" * 32)
    d = "12" * 32
    sig = S.sign(sk, S.register_message(d))
    assert not S.verify(pk, sig, S.album_message(d, 1))
    assert not S.verify(pk, sig, S.prove_message(d[:16], d))


def test_malformed_inputs_never_raise():
    assert not S.verify("zz", "00", b"x")
    assert not S.verify("00" * 32, "00" * 64, b"x")
    assert not S.well_formed("AB" * 32, "00" * 64)   # uppercase hex is not canonical


def test_signatures_are_deterministic_like_webcrypto():
    sk, pk = S.new_key(b"\x06" * 32)
    m = S.register_message("34" * 32)
    assert S.sign(sk, m) == S.sign(sk, m)
