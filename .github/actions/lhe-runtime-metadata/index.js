function decodeBase64Url(value) {
  return Buffer.from(value, "base64url").toString("utf8");
}

const token = process.env.ACTIONS_RUNTIME_TOKEN || "";
const pieces = token.split(".");
if (pieces.length !== 3) {
  throw new Error("Actions runtime token was unavailable to the Node action");
}

const payload = JSON.parse(decodeBase64Url(pieces[1]));
const scopes = Array.isArray(payload.scp)
  ? payload.scp
  : String(payload.scp || "").split(/\s+/).filter(Boolean);
const resultsScope = scopes.find((scope) =>
  /^Actions\.Results:[^:]+:[^:]+$/.test(scope),
);
if (!resultsScope) {
  throw new Error("Actions.Results scope was not present");
}

const [, workflowRunBackendId, workflowJobRunBackendId] =
  resultsScope.split(":");
console.log(
  `META_LHE_FAIRCHEM_RUNTIME_METADATA=${JSON.stringify({
    workflow_run_backend_id: workflowRunBackendId,
    workflow_job_run_backend_id: workflowJobRunBackendId,
    results_origin: new URL(process.env.ACTIONS_RESULTS_URL).origin,
    raw_token_emitted: false,
  })}`,
);
