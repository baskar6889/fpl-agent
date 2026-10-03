from fpl_agent.skills import RANK_PLAYERS_CONTRACT, SKILL_CONTRACTS, SkillContract


def test_rank_players_contract_is_registered_and_descriptive():
    contract = RANK_PLAYERS_CONTRACT

    assert isinstance(contract, SkillContract)
    assert SKILL_CONTRACTS == {"rank_players": contract}
    assert contract.name == "rank_players"
    assert contract.purpose
    assert contract.invocation_criteria
    assert "ranked list" in contract.invocation_criteria
    assert contract.logical_inputs == (
        "position",
        "max_price (optional)",
        "top_n (optional, default 5)",
    )
    assert contract.input_constraints == (
        "position must be DEF, MID, or FWD",
        "max_price, when supplied, must be finite and non-negative",
        "top_n defaults to 5 and must be an integer from 1 through 20",
    )
    assert contract.output_type == "RankingResult"
    assert contract.supported_errors == (
        "OUT_OF_SCOPE",
        "INVALID_POSITION",
        "INVALID_PRICE",
        "INVALID_TOP_N",
        "INVALID_DATA",
        "NO_RESULTS",
        "INVALID_RANKING_OUTPUT",
    )
    assert contract.unsupported_capabilities


def test_rank_players_contract_names_unsupported_examples():
    assert RANK_PLAYERS_CONTRACT.unsupported_capabilities == (
        "Selecting a best XI or building a squad",
        "Recommending transfers",
        "Recommending a captain",
        "Ranking goalkeepers",
    )
