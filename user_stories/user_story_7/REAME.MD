
### User Story 7: User provides safemodel with no data

In this example, a user builds a model using the SafeModel class, and wraps their data in a Target object. However, the researcher forgets to call request_release() or Target.save(), which prevents any useful information regarding training data or model to the TRE.

Because of this, we are unable to proceed with this release, and the user is requested to call one of the above functions.
