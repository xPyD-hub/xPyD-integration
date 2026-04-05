# Add this to each sub-project's .github/workflows/ci.yml
# as an additional job that triggers integration tests.

# Template — copy to xPyD-sim, xPyD-proxy, xPyD-bench CI workflows:

  integration-trigger:
    needs: [test]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Trigger integration tests
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh api repos/xPyD-hub/xPyD-integration/dispatches \
            -f event_type=dependency-updated \
            -f "client_payload[repo]=${{ github.repository }}" \
            -f "client_payload[sha]=${{ github.sha }}"
