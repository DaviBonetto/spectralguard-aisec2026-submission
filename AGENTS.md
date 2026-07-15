# AGENTS.md — SpectralGuard → ACM AISec 2026

## Missão
- Transformar o paper SpectralGuard numa submissão tecnicamente sólida para ACM AISec 2026.
- Deadline: 24 de julho de 2026, 23:59 AoE. Formato: 10 páginas de corpo + até 2 de apêndice, ACM sigconf, double-blind.

## Papéis
- Tesla (Codex): orquestra, opera neste computador e decide delegação.
- Newton: paralelismo ou verificação independente de qualquer item P0; Tesla fornece os prompts.
- Alfred: consultor estratégico e teórico via humano; não executa diretamente.

## Fonte da verdade
- `plano/PLANO-MESTRE-AISec2026.md`: fases 0–13, tarefas e verificações.
- `plano/spectralguard-aisec2026-battle-plan.md`: diagnóstico técnico detalhado.
- Leia ambos integralmente antes de alterar código. Toda ação deve referenciar fase/item.

## Regras de ouro
1. Todo arquivo precisa ser referenciado por script, README ou apêndice de reprodutibilidade.
2. Nenhum item está concluído sem a evidência definida em sua verificação.
3. `plano/` nunca é rastreado por Git; se aparecer no status, pare e corrija.
4. Double-blind é inegociável: nada de nome, e-mail, afiliação ou pista de identidade no paper ou repo público.
5. Atualize `plano/arvore.md` após mudança estrutural relevante.
6. Registre sempre seeds, versões e hashes de revisão de modelos.
7. Quando só o humano puder fornecer algo, peça explicitamente; não invente nem pule.
8. Newton é para paralelismo/verificação; Alfred é para julgamento estratégico via humano.

## Política do paper
- Escreva estrutura, teoria e metodologia em `paper/` progressivamente desde a Fase 0.
- Abstract, título, a alegação central sobre SSMs/Transformers e conclusão permanecem como placeholders até Fase 3.1 e Fase 7.
- Não escreva como resultado final nada que dependa de experimentos ainda não concluídos.

## Dependência conhecida
- Fase 1, item 1.2 e Fase 3, item 3.16 exigem CLASP (arXiv:2603.12206). Busque-o publicamente; se indisponível, peça ao humano.

## Restrições
- Sessões de GPU: no máximo 5 h consecutivas por conta.
- Após qualquer edição em `paper/`, verifique o orçamento de 10+2 páginas.
- Nunca delete fora de `_archive_old/` ou dentro de `plano/` sem confirmação explícita.
